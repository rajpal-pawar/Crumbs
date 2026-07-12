import { useState, useEffect, useRef, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';

export interface DirEntry {
  path: string;
  state: 'queued' | 'scanning' | 'indexing' | 'completed';
}

interface ProgressPayload {
  status: string;
  indexed: number;
  total: number;
  directories?: DirEntry[];
}

interface DaemonStatusResponse {
  paused: boolean;
  db_size: number;
  onnx_memory: number;
  doc_count: number;
  watch_dirs: string[];
  embed_batch_size: number;
  onnx_threads: number;
}

// Middle-truncate path helper
function truncatePath(path: string, maxLen = 45): string {
  if (!path) return '';
  if (path.length <= maxLen) return path;
  const half = Math.floor((maxLen - 3) / 2);
  return path.slice(0, half) + '…' + path.slice(-half);
}

// Debounce helper
function useDebouncedCallback<T extends (...args: any[]) => void>(
  fn: T,
  delayMs: number,
): T {
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const fnRef = useRef(fn);
  fnRef.current = fn;

  return useCallback(((...args: any[]) => {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => fnRef.current(...args), delayMs);
  }) as unknown as T, [delayMs]);
}

export default function Dashboard() {
  const [dirs, setDirs] = useState<DirEntry[]>([]);
  const [indexed, setIndexed] = useState(0);
  const [total, setTotal] = useState(0);
  const [status, setStatus] = useState('idle');
  const [paused, setPaused] = useState(false);
  const [docCount, setDocCount] = useState(0);
  
  const [dbSize, setDbSize] = useState(0);
  const [onnxMemory, setOnnxMemory] = useState(0);
  
  const [config, setConfig] = useState({ batchSize: 5, threads: 2, indexParallelism: 1 });
  const [managedFolders, setManagedFolders] = useState<string[]>([]);
  const [folderUpdating, setFolderUpdating] = useState(false);

  // 1. Fetch daemon status, metrics, and folders on mount and poll every 2.5 seconds
  const fetchStatus = useCallback(async () => {
    try {
      const res = await invoke<DaemonStatusResponse>('status');
      if (res) {
        setPaused(!!res.paused);
        setDbSize(res.db_size || 0);
        setOnnxMemory(res.onnx_memory || 0);
        setDocCount(res.doc_count || 0);
        setManagedFolders(res.watch_dirs || []);
        setConfig(prev => ({
          ...prev,
          batchSize: res.embed_batch_size || prev.batchSize,
          threads: res.onnx_threads || prev.threads,
        }));
      }
    } catch (err) {
      console.error('[Dashboard] Failed to fetch daemon status:', err);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 2500);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  // 2. Listen to live reindex progress events
  useEffect(() => {
    let unlisten: (() => void) | undefined;
    listen<ProgressPayload>('crumbs://index-progress', (event) => {
      const p = event.payload;
      if (!p) return;
      if (p.indexed !== undefined) setIndexed(p.indexed);
      if (p.total !== undefined) setTotal(p.total);
      if (p.status) setStatus(p.status);
      if (p.directories && p.directories.length > 0) {
        setDirs(p.directories);
      }
    }).then((un) => {
      unlisten = un;
    });

    return () => {
      if (unlisten) unlisten();
    };
  }, []);

  // 3. Debounced config updater for engine sliders
  const sendConfig = useDebouncedCallback((newConfig: { batchSize: number; threads: number; paused: boolean }) => {
    invoke('update_engine_config', {
      batchSize: newConfig.batchSize,
      threads: newConfig.threads,
      paused: newConfig.paused,
    }).catch((err) => console.error('[Dashboard] Config update failed:', err));
  }, 400);

  const handleBatchSizeChange = (val: number) => {
    const next = { ...config, batchSize: val };
    setConfig(next);
    sendConfig({ batchSize: val, threads: config.threads, paused });
  };

  const handleThreadsChange = (val: number) => {
    const next = { ...config, threads: val };
    setConfig(next);
    sendConfig({ batchSize: config.batchSize, threads: val, paused });
  };

  const handleIndexParallelismChange = (val: number) => {
    setConfig(prev => ({ ...prev, indexParallelism: val }));
  };

  // 4. Pause / Resume Engine handler
  const handleTogglePause = async () => {
    const nextPaused = !paused;
    setPaused(nextPaused);
    try {
      await invoke('update_engine_config', {
        batchSize: config.batchSize,
        threads: config.threads,
        paused: nextPaused,
      });
    } catch (err) {
      console.error('[Dashboard] Failed to toggle pause:', err);
    }
  };

  // 5. Folder CRUD controls
  const handleAddFolder = async () => {
    try {
      const paths = await invoke<string[]>('select_folders_dialog');
      if (paths && paths.length > 0) {
        const combined = [...managedFolders];
        for (const p of paths) {
          if (!combined.includes(p)) {
            combined.push(p);
          }
        }
        setFolderUpdating(true);
        await invoke('update_monitored_folders', { folders: combined, isOnboarded: true });
        setManagedFolders(combined);
        setFolderUpdating(false);
        fetchStatus();
      }
    } catch (err) {
      console.error('[Dashboard] Add folder failed:', err);
      setFolderUpdating(false);
    }
  };

  const handleRemoveFolder = async (path: string) => {
    const updated = managedFolders.filter(p => p !== path);
    setFolderUpdating(true);
    try {
      await invoke('update_monitored_folders', { folders: updated, isOnboarded: true });
      setManagedFolders(updated);
      fetchStatus();
    } catch (err) {
      console.error('[Dashboard] Remove folder failed:', err);
    }
    setFolderUpdating(false);
  };

  // Format bytes to human readable sizes
  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  const isEngineActive = (status === 'indexing' || status === 'scanning') && !paused;
  const pct = total > 0 ? Math.round((indexed / total) * 100) : 0;  return (
    <div className="min-h-screen bg-[#070709] text-zinc-100 font-sans p-8 flex flex-col gap-6 selection:bg-indigo-500/30 relative overflow-hidden z-0">
      
      {/* Decorative ambient background glow blobs for glassmorphism */}
      <div className="absolute top-[-10%] right-[-15%] w-[60vw] h-[60vw] rounded-full bg-indigo-600/10 blur-[130px] pointer-events-none -z-10" />
      <div className="absolute bottom-[-15%] left-[-15%] w-[60vw] h-[60vw] rounded-full bg-violet-600/8 blur-[130px] pointer-events-none -z-10" />
      <div className="absolute top-[35%] left-[20%] w-[40vw] h-[40vw] rounded-full bg-fuchsia-600/4 blur-[120px] pointer-events-none -z-10" />

      {/* ── SECTION 1: HEADER BLOCK (Global Telemetry) ── */}
      <header className="bg-zinc-900/35 backdrop-blur-xl border border-white/[0.04] rounded-2xl shadow-[0_8px_32px_0_rgba(0,0,0,0.37)] p-6 flex flex-col md:flex-row md:items-center justify-between gap-6 relative overflow-hidden group/card hover:border-white/[0.08] transition-all duration-300">
        <div className="flex-1 flex flex-col gap-2">
          <div className="flex items-center gap-3">
            <div className={`w-3 h-3 rounded-full ${paused ? 'bg-zinc-500 shadow-[0_0_12px_rgba(113,113,122,0.5)]' : isEngineActive ? 'bg-indigo-500 shadow-[0_0_15px_rgba(99,102,241,0.8)] animate-pulse' : 'bg-zinc-650 shadow-[0_0_12px_rgba(113,113,122,0.3)]'}`} />
            <h1 className="text-xl font-bold tracking-tight text-white">Crumbs Engine Telemetry</h1>
            <span className="text-[10px] font-sans font-semibold px-2 py-0.5 rounded bg-white/5 border border-white/10 text-zinc-400 uppercase tracking-wider">
              {paused ? 'PAUSED' : isEngineActive ? 'CRAWLING' : 'IDLE'}
            </span>
          </div>
          
          <div className="flex justify-between items-center text-xs text-zinc-450 mt-2">
            <span>Progress: <span className="font-mono text-zinc-200 font-semibold">{indexed.toLocaleString()}</span> / <span className="font-mono text-zinc-200 font-semibold">{total.toLocaleString()}</span> files indexed</span>
            <span className="font-mono text-indigo-400 font-bold">{pct}%</span>
          </div>
          
          {/* Glass-styled Neon gradient progress bar */}
          <div className="w-full bg-zinc-950/60 backdrop-blur-sm rounded-full h-2.5 overflow-hidden border border-white/[0.03]">
            <div
              className={`h-full rounded-full bg-gradient-to-r from-indigo-500 via-violet-500 to-fuchsia-500 transition-all duration-500 ease-out shadow-[0_0_12px_rgba(139,92,246,0.3)] ${isEngineActive ? 'animate-pulse' : ''}`}
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>

        {/* Toggle Pause Switch */}
        <button
          onClick={handleTogglePause}
          className={`px-5 py-2.5 rounded-xl font-semibold border text-xs tracking-wider uppercase transition-all duration-300 flex items-center gap-2 cursor-pointer active:scale-[0.98] ${
            paused
              ? 'bg-zinc-800/40 text-zinc-300 border-white/[0.04] hover:bg-zinc-800/70 hover:border-white/[0.08] hover:text-white'
              : 'bg-indigo-500/10 text-indigo-300 border-indigo-500/20 hover:bg-indigo-500/20 hover:border-indigo-500/40 hover:shadow-[0_0_15px_rgba(99,102,241,0.15)]'
          }`}
        >
          {paused ? (
            <>
              <svg className="w-3.5 h-3.5 fill-current" viewBox="0 0 24 24">
                <path d="M8 5v14l11-7z" />
              </svg>
              Resume Engine
            </>
          ) : (
            <>
              <svg className="w-3.5 h-3.5 fill-current" viewBox="0 0 24 24">
                <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
              </svg>
              Pause Engine
            </>
          )}
        </button>
      </header>

      {/* Bento Grid Panel Layout */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 auto-rows-[250px] md:auto-rows-[220px]">
        
        {/* ── SECTION 2: DIRECTORY MATRIX (Large Left-Column Tile) ── */}
        <section className="bg-zinc-900/35 backdrop-blur-xl border border-white/[0.04] rounded-2xl shadow-[0_8px_32px_0_rgba(0,0,0,0.37)] p-6 md:col-span-2 md:row-span-2 flex flex-col justify-between hover:border-white/[0.08] transition-all duration-300 group/card">
          <div className="flex flex-col gap-4 overflow-hidden h-full">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <svg className="w-4 h-4 text-zinc-400" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                </svg>
                <h2 className="text-zinc-400 text-sm font-semibold tracking-wide uppercase my-auto">Monitored Folder Matrix</h2>
              </div>
              <button
                onClick={handleAddFolder}
                disabled={folderUpdating}
                className="text-xs bg-white/5 border border-white/10 hover:bg-white/10 hover:border-white/25 text-indigo-400 px-3 py-1.5 rounded-lg flex items-center gap-1.5 cursor-pointer disabled:opacity-50 transition-all active:scale-[0.97]"
              >
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
                </svg>
                Add Folder
              </button>
            </div>

            {/* Matrix grid list of folders */}
            <div className="flex-1 overflow-y-auto pr-1 flex flex-col gap-2 mt-2">
              {managedFolders.length === 0 ? (
                <div className="flex-1 flex items-center justify-center border border-dashed border-white/10 rounded-xl p-6 text-zinc-500 text-xs bg-zinc-950/20">
                  No folders registered. Click 'Add Folder' to start mapping directory files.
                </div>
              ) : (
                managedFolders.map((path) => {
                  const pathLower = path.toLowerCase();
                  const dirEntry = dirs.find(d => pathLower.includes(d.path.replace(/^~\//, '').toLowerCase()) || d.path.toLowerCase().includes(pathLower));
                  const state = dirEntry ? dirEntry.state : 'completed';

                  return (
                    <div key={path} className="flex items-center justify-between p-3 rounded-xl bg-zinc-950/30 border border-white/[0.02] hover:bg-zinc-950/60 hover:border-white/[0.06] transition-all duration-200 group/item">
                      <div className="flex flex-col gap-0.5 min-w-0">
                        <span className="font-mono text-xs text-zinc-300 truncate" title={path}>
                          {truncatePath(path)}
                        </span>
                        <div className="flex items-center gap-2">
                          <span className={`text-[10px] uppercase font-bold tracking-wider ${
                            state === 'scanning' ? 'text-amber-400 animate-pulse' :
                            state === 'indexing' ? 'text-indigo-400 animate-pulse' :
                            state === 'queued' ? 'text-zinc-500' : 'text-emerald-400 font-semibold'
                          }`}>
                            {state}
                          </span>
                        </div>
                      </div>
                      
                      <button
                        onClick={() => handleRemoveFolder(path)}
                        disabled={folderUpdating}
                        className="opacity-0 group-hover/item:opacity-100 text-zinc-500 hover:text-red-400 p-1.5 hover:bg-red-500/10 rounded-lg transition-all duration-250 cursor-pointer"
                        title="Remove Folder"
                      >
                        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                      </button>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </section>

        {/* ── SECTION 3: LIVE THREAD VISUALIZER (Center Tile) ── */}
        <section className="bg-zinc-900/35 backdrop-blur-xl border border-white/[0.04] rounded-2xl shadow-[0_8px_32px_0_rgba(0,0,0,0.37)] p-6 flex flex-col justify-between relative overflow-hidden hover:border-white/[0.08] transition-all duration-300 group/card">
          <div className="flex items-center gap-2">
            <svg className="w-4 h-4 text-zinc-400" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            <h2 className="text-zinc-400 text-sm font-semibold tracking-wide uppercase">Live CPU Threads</h2>
          </div>

          {/* Central CSS graphical visualizer */}
          <div className="flex-1 flex items-center justify-center my-4 relative">
            
            {/* Core engine visualizer */}
            <div className={`w-12 h-12 rounded-full flex items-center justify-center bg-zinc-950/60 border border-white/5 backdrop-blur-md z-10 relative ${isEngineActive ? 'shadow-[0_0_20px_rgba(99,102,241,0.2)] animate-pulse' : ''}`}>
              <div className={`w-3.5 h-3.5 rounded-full ${paused ? 'bg-zinc-500 shadow-[0_0_12px_rgba(113,113,122,0.4)]' : isEngineActive ? 'bg-indigo-500 shadow-[0_0_15px_rgba(99,102,241,0.8)] animate-ping' : 'bg-zinc-650 shadow-[0_0_12px_rgba(113,113,122,0.2)]'}`} />
            </div>

            {/* Orbiting slots visualizer */}
            <div className="absolute inset-0 flex items-center justify-center">
              {Array.from({ length: config.threads || 2 }).map((_, i, arr) => {
                const angle = (360 / arr.length) * i;
                const distance = 42; // px from center
                const x = Math.cos((angle * Math.PI) / 180) * distance;
                const y = Math.sin((angle * Math.PI) / 180) * distance;
                
                return (
                  <div
                    key={i}
                    style={{
                      transform: `translate(${x}px, ${y}px)`,
                      transition: 'transform 0.5s ease',
                    }}
                    className={`absolute w-5 h-5 rounded-full bg-zinc-950/70 border text-[9px] font-sans font-semibold flex items-center justify-center backdrop-blur-sm ${
                      paused
                        ? 'border-white/5 text-zinc-600'
                        : isEngineActive
                        ? 'border-indigo-500/60 text-indigo-300 shadow-[0_0_10px_rgba(99,102,241,0.2)] animate-pulse'
                        : 'border-white/5 text-zinc-500'
                    }`}
                  >
                    T{i + 1}
                  </div>
                );
              })}
            </div>

            {/* Ambient rotating rings when engine is active */}
            {isEngineActive && (
              <>
                <div className="absolute w-24 h-24 border border-dashed border-indigo-500/20 rounded-full animate-spin [animation-duration:6s]" />
                <div className="absolute w-28 h-28 border border-dashed border-violet-500/10 rounded-full animate-spin [animation-duration:8s] [animation-direction:reverse]" />
              </>
            )}
          </div>

          <div className="text-center font-sans text-[10px] text-zinc-500 uppercase tracking-wider">
            {isEngineActive ? `Indexing batches with ${config.threads} thread workers` : paused ? 'Thread pool suspended' : 'Thread workers idle'}
          </div>
        </section>

        {/* ── SECTION 4: SYSTEM METRICS TILE (Right Column - Small) ── */}
        <section className="bg-zinc-900/35 backdrop-blur-xl border border-white/[0.04] rounded-2xl shadow-[0_8px_32px_0_rgba(0,0,0,0.37)] p-6 flex flex-col justify-between hover:border-white/[0.08] transition-all duration-300 group/card">
          <div className="flex items-center gap-2">
            <svg className="w-4 h-4 text-zinc-400" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2m0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 002 2h2a2 2 0 002-2z" />
            </svg>
            <h2 className="text-zinc-400 text-sm font-semibold tracking-wide uppercase">System Metrics</h2>
          </div>

          <div className="flex flex-col gap-3 my-2">
            {/* Database Size metric */}
            <div className="flex justify-between items-center p-3 rounded-xl bg-zinc-950/30 border border-white/[0.02]">
              <div className="flex flex-col">
                <span className="text-[10px] font-sans font-semibold text-zinc-500 uppercase tracking-wider">Database Size</span>
                <span className="text-base font-bold text-white font-mono">{formatBytes(dbSize)}</span>
              </div>
              <svg className="w-5 h-5 text-zinc-600" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M4 7v10c0 2 1.5 3 3.5 3h9c2 0 3.5-1 3.5-3V7c0-2-1.5-3-3.5-3h-9C5.5 4 4 5 4 7zm0 0c0 1 1.5 1.5 3.5 1.5h9c2 0 3.5-.5 3.5-1.5M4 12c0 1 1.5 1.5 3.5 1.5h9c2 0 3.5-.5 3.5-1.5" />
              </svg>
            </div>

            {/* ONNX Model Cache footprint */}
            <div className="flex justify-between items-center p-3 rounded-xl bg-zinc-950/30 border border-white/[0.02]">
              <div className="flex flex-col">
                <span className="text-[10px] font-sans font-semibold text-zinc-500 uppercase tracking-wider">Active ONNX RAM</span>
                <span className="text-base font-bold text-white font-mono">{formatBytes(onnxMemory)}</span>
              </div>
              <svg className="w-5 h-5 text-zinc-600" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
              </svg>
            </div>
          </div>

          <div className="font-sans text-[10px] text-zinc-400 flex justify-between">
            <span>Indexed Documents:</span>
            <span className="text-indigo-400 font-bold font-mono">{docCount.toLocaleString()}</span>
          </div>
        </section>

        {/* ── SECTION 5: ENGINE TUNING SLIDERS (Bottom Tile) ── */}
        <section className="bg-zinc-900/35 backdrop-blur-xl border border-white/[0.04] rounded-2xl shadow-[0_8px_32px_0_rgba(0,0,0,0.37)] p-6 md:col-span-3 flex flex-col justify-between hover:border-white/[0.08] transition-all duration-300 group/card">
          <div className="flex items-center gap-2">
            <svg className="w-4 h-4 text-zinc-400" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
            </svg>
            <h2 className="text-zinc-400 text-sm font-semibold tracking-wide uppercase">Engine Parameter Tuning</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 my-4">
            
            {/* Index Parallelism Slider */}
            <div className="flex flex-col gap-2">
              <div className="flex justify-between items-center">
                <label className="text-xs text-zinc-400 font-semibold uppercase tracking-wider" htmlFor="parallelism-slider">Index Parallelism</label>
                <span className="text-xs font-mono font-bold text-indigo-400 px-2 py-0.5 rounded bg-zinc-950/60 border border-white/5 backdrop-blur-sm">{config.indexParallelism}</span>
              </div>
              <input
                id="parallelism-slider"
                type="range"
                min="1"
                max="8"
                step="1"
                value={config.indexParallelism}
                onChange={(e) => handleIndexParallelismChange(parseInt(e.target.value))}
                className="w-full accent-indigo-500 bg-zinc-950/60 border border-white/[0.03] backdrop-blur-sm h-2 rounded-lg appearance-none cursor-pointer hover:border-white/[0.08] transition-all"
              />
              <div className="flex justify-between text-[10px] text-zinc-550 px-0.5 font-mono">
                <span>1</span>
                <span>8</span>
              </div>
            </div>

            {/* CPU Threads Slider */}
            <div className="flex flex-col gap-2">
              <div className="flex justify-between items-center">
                <label className="text-xs text-zinc-400 font-semibold uppercase tracking-wider" htmlFor="threads-slider">CPU Threads</label>
                <span className="text-xs font-mono font-bold text-indigo-400 px-2 py-0.5 rounded bg-zinc-950/60 border border-white/5 backdrop-blur-sm">{config.threads}</span>
              </div>
              <input
                id="threads-slider"
                type="range"
                min="1"
                max="16"
                step="1"
                value={config.threads}
                onChange={(e) => handleThreadsChange(parseInt(e.target.value))}
                className="w-full accent-indigo-500 bg-zinc-950/60 border border-white/[0.03] backdrop-blur-sm h-2 rounded-lg appearance-none cursor-pointer hover:border-white/[0.08] transition-all"
              />
              <div className="flex justify-between text-[10px] text-zinc-550 px-0.5 font-mono">
                <span>1</span>
                <span>16</span>
              </div>
            </div>

            {/* Embedding Batch Size Slider */}
            <div className="flex flex-col gap-2">
              <div className="flex justify-between items-center">
                <label className="text-xs text-zinc-400 font-semibold uppercase tracking-wider" htmlFor="batch-slider">Embedding Batch Size</label>
                <span className="text-xs font-mono font-bold text-indigo-400 px-2 py-0.5 rounded bg-zinc-950/60 border border-white/5 backdrop-blur-sm">{config.batchSize}</span>
              </div>
              <input
                id="batch-slider"
                type="range"
                min="1"
                max="50"
                step="1"
                value={config.batchSize}
                onChange={(e) => handleBatchSizeChange(parseInt(e.target.value))}
                className="w-full accent-indigo-500 bg-zinc-950/60 border border-white/[0.03] backdrop-blur-sm h-2 rounded-lg appearance-none cursor-pointer hover:border-white/[0.08] transition-all"
              />
              <div className="flex justify-between text-[10px] text-zinc-550 px-0.5 font-mono">
                <span>1</span>
                <span>50</span>
              </div>
            </div>

          </div>

          <div className="text-[10px] text-zinc-500 italic text-center font-sans">
            Parameter changes are debounced and applied live to the MPSC indexing consumer without restart.
          </div>
        </section>

      </div>

      <footer className="text-center text-[10px] font-sans text-zinc-650 tracking-widest mt-4 uppercase">
        CRUMBS BACKGROUND PROCESSOR • VERSION 0.1.0
      </footer>
    </div>
  );
}
