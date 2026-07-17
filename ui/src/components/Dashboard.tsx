import { useState, useEffect, useRef, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { ThemeToggle } from '../ThemeContext';

export interface DirEntry {
  path: string;
  state: 'queued' | 'scanning' | 'indexing' | 'completed';
}

interface ProgressPayload {
  status: string;
  indexed: number;
  processed?: number;
  errors?: number;
  skipped?: number;
  total: number;
  directories?: DirEntry[];
  failed_files?: FileIssue[];
  skipped_files?: FileIssue[];
}

interface FileIssue {
  path: string;
  reason: string;
}

interface IndexedFile {
  path: string;
  title: string;
  mime_type: string;
  size_bytes: number;
}

interface DaemonStatusResponse {
  paused: boolean;
  status: string;
  db_size: number;
  onnx_memory: number;
  doc_count: number;
  watch_dirs: string[];
  embed_batch_size: number;
  onnx_threads: number;
  failed_files?: FileIssue[];
  skipped_files?: FileIssue[];
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
  const [processed, setProcessed] = useState(0);
  const [errors, setErrors] = useState(0);
  const [skipped, setSkipped] = useState(0);
  const [total, setTotal] = useState(0);
  const [status, setStatus] = useState('idle');
  const statusRef = useRef('idle');
  const [paused, setPaused] = useState(false);
  const [docCount, setDocCount] = useState(0);
  
  const [dbSize, setDbSize] = useState(0);
  const [onnxMemory, setOnnxMemory] = useState(0);
  
  const [config, setConfig] = useState({ batchSize: 5, threads: 2, indexParallelism: 1 });
  const [managedFolders, setManagedFolders] = useState<string[]>([]);
  const [folderUpdating, setFolderUpdating] = useState(false);
  const [failedFiles, setFailedFiles] = useState<FileIssue[]>([]);
  const [skippedFiles, setSkippedFiles] = useState<FileIssue[]>([]);
  const [fileInspectorTab, setFileInspectorTab] = useState<'indexed' | 'failed' | 'skipped' | null>(null);
  const [indexedFiles, setIndexedFiles] = useState<IndexedFile[]>([]);
  const [indexedFilesLoading, setIndexedFilesLoading] = useState(false);

  // 1. Fetch daemon status, metrics, and folders on mount and poll every 2.5 seconds
  const fetchStatus = useCallback(async () => {
    try {
      const res = await invoke<DaemonStatusResponse>('status');
      if (res) {
        setPaused(!!res.paused);
        if (res.status) {
          setStatus(res.status);
          statusRef.current = res.status;
        }
        setDbSize(res.db_size || 0);
        setOnnxMemory(res.onnx_memory || 0);
        setDocCount(res.doc_count || 0);
        setManagedFolders(res.watch_dirs || []);
        setConfig(prev => ({
          ...prev,
          batchSize: res.embed_batch_size || prev.batchSize,
          threads: res.onnx_threads || prev.threads,
        }));
        if (res.failed_files && statusRef.current !== 'indexing') setFailedFiles(res.failed_files);
        if (res.skipped_files && statusRef.current !== 'indexing') setSkippedFiles(res.skipped_files);
        
        if (res.status !== 'indexing' && res.status !== 'scanning') {
          const e = res.failed_files?.length || 0;
          const s = res.skipped_files?.length || 0;
          const i = res.doc_count || 0;
          const t = i + e + s;
          setIndexed(i);
          setErrors(e);
          setSkipped(s);
          setTotal(t);
          setProcessed(t);
        }
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
      if (p.processed !== undefined) setProcessed(p.processed);
      if (p.errors !== undefined) setErrors(p.errors || 0);
      if (p.skipped !== undefined) setSkipped(p.skipped || 0);
      if (p.total !== undefined) setTotal(p.total);
      if (p.status) {
        setStatus(p.status);
        statusRef.current = p.status;
      }
      if (p.directories && p.directories.length > 0) {
        setDirs(p.directories);
      }
      if (p.failed_files) setFailedFiles(p.failed_files);
      if (p.skipped_files) setSkippedFiles(p.skipped_files);
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
        try {
          await invoke('update_monitored_folders', { folders: combined, isOnboarded: true });
          setManagedFolders(combined);
          fetchStatus();
        } catch (error) {
          alert(error);
        }
        setFolderUpdating(false);
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
    } catch (error) {
      alert(error);
    }
    setFolderUpdating(false);
  };

  // Format bytes to human readable sizes
  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  const isEngineActive = (status === 'indexing' || status === 'scanning') && !paused;
  const effectiveProcessed = Math.max(processed, indexed + errors + skipped);
  const pct = total > 0 ? Math.min(100, Math.round((effectiveProcessed / total) * 100)) : 0;

  // CSS classes for glass cards
  const glassCard = "glass-card";
  const sectionTitle = "dashboard-section__title";

  return (
    <div style={{
      height: '100vh',
      background: 'var(--c-bg)',
      color: 'var(--c-text)',
      fontFamily: "'Inter', system-ui, sans-serif",
      padding: '24px',
      display: 'flex',
      flexDirection: 'column',
      gap: '20px',
      position: 'relative',
      overflowX: 'hidden',
      overflowY: 'auto',
      pointerEvents: 'auto',
    }}>
      
      {/* Ambient glass blobs */}
      <div style={{
        position: 'absolute', top: '-10%', right: '-15%',
        width: '60vw', height: '60vw', borderRadius: '50%',
        background: 'rgba(224,168,96,0.06)', filter: 'blur(130px)',
        pointerEvents: 'none', zIndex: 0,
      }} />
      <div style={{
        position: 'absolute', bottom: '-15%', left: '-15%',
        width: '60vw', height: '60vw', borderRadius: '50%',
        background: 'rgba(212,136,106,0.04)', filter: 'blur(130px)',
        pointerEvents: 'none', zIndex: 0,
      }} />

      {/* HEADER — Global Telemetry */}
      <header className={glassCard} style={{ padding: '24px', position: 'relative', zIndex: 1 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '16px', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <img src="/logo.png" alt="Crumbs" style={{ width: '40px', height: '40px', objectFit: 'contain' }} />
            <div style={{
              width: '10px', height: '10px', borderRadius: '50%',
              background: paused ? '#71717a' : isEngineActive ? 'var(--c-accent)' : '#52525b',
              boxShadow: isEngineActive ? '0 0 14px rgba(224,168,96,0.7)' : 'none',
              animation: isEngineActive ? 'pulse 1.5s ease infinite' : 'none',
            }} />
            <h1 style={{ fontSize: '18px', fontWeight: 700, letterSpacing: '-0.02em' }}>Crumbs Engine</h1>
            <span style={{
              fontSize: '10px', fontWeight: 600, padding: '2px 8px', borderRadius: '6px',
              background: 'var(--c-hit-bg)', border: '1px solid var(--c-border)',
              color: 'var(--c-text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em',
            }}>
              {paused ? 'PAUSED' : isEngineActive ? 'CRAWLING' : 'IDLE'}
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <ThemeToggle />
            <button onClick={handleTogglePause} style={{
              padding: '8px 16px', borderRadius: '10px', fontWeight: 600, fontSize: '11px',
              border: '1px solid var(--c-border)', cursor: 'pointer',
              background: paused ? 'var(--c-hit-bg)' : 'rgba(224,168,96,0.1)',
              color: paused ? 'var(--c-text-muted)' : 'var(--c-accent)',
              textTransform: 'uppercase', letterSpacing: '0.06em',
              display: 'flex', alignItems: 'center', gap: '6px',
              transition: 'all 200ms ease',
            }}>
              {paused ? (
                <><svg style={{width:'12px',height:'12px',fill:'currentColor'}} viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>Resume</>
              ) : (
                <><svg style={{width:'12px',height:'12px',fill:'currentColor'}} viewBox="0 0 24 24"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>Pause</>
              )}
            </button>
          </div>
        </div>

        {/* Progress */}
        <div style={{ marginTop: '16px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: 'var(--c-text-muted)', marginBottom: '6px' }}>
            <span>
              <span style={{ color: 'var(--c-text)', fontWeight: 600, fontFamily: 'monospace' }}>{effectiveProcessed.toLocaleString()}</span> / <span style={{ color: 'var(--c-text)', fontWeight: 600, fontFamily: 'monospace' }}>{total.toLocaleString()}</span> files
              {effectiveProcessed > 0 && <span style={{ display: 'inline-flex', gap: '6px', marginLeft: '6px' }}>
                (<button
                  onClick={() => {
                    if (fileInspectorTab === 'indexed') { setFileInspectorTab(null); return; }
                    setFileInspectorTab('indexed');
                    setIndexedFilesLoading(true);
                    invoke<{ documents: IndexedFile[]; total: number }>('list_indexed_files')
                      .then(res => setIndexedFiles(res.documents || []))
                      .catch(err => console.error('[Dashboard] list_indexed_files failed:', err))
                      .finally(() => setIndexedFilesLoading(false));
                  }}
                  style={{
                    background: 'none', border: 'none', padding: 0, cursor: 'pointer',
                    color: fileInspectorTab === 'indexed' ? 'var(--c-accent)' : '#4ade80',
                    fontFamily: 'inherit', fontSize: 'inherit', textDecoration: 'underline',
                    textDecorationStyle: 'dotted', textUnderlineOffset: '2px',
                  }}
                >{indexed.toLocaleString()} indexed</button>
                {errors > 0 && <>, <button
                  onClick={() => setFileInspectorTab(fileInspectorTab === 'failed' ? null : 'failed')}
                  style={{
                    background: 'none', border: 'none', padding: 0, cursor: 'pointer',
                    color: fileInspectorTab === 'failed' ? 'var(--c-accent)' : '#f87171',
                    fontFamily: 'inherit', fontSize: 'inherit', textDecoration: 'underline',
                    textDecorationStyle: 'dotted', textUnderlineOffset: '2px',
                  }}
                >{errors.toLocaleString()} failed</button></>}
                {skipped > 0 && <>, <button
                  onClick={() => setFileInspectorTab(fileInspectorTab === 'skipped' ? null : 'skipped')}
                  style={{
                    background: 'none', border: 'none', padding: 0, cursor: 'pointer',
                    color: fileInspectorTab === 'skipped' ? 'var(--c-accent)' : '#facc15',
                    fontFamily: 'inherit', fontSize: 'inherit', textDecoration: 'underline',
                    textDecorationStyle: 'dotted', textUnderlineOffset: '2px',
                  }}
                >{skipped.toLocaleString()} skipped</button></>}
                )
              </span>}
            </span>
            <span style={{ color: 'var(--c-accent)', fontWeight: 700, fontFamily: 'monospace' }}>{pct}%</span>
          </div>
          <div className="dashboard-progress__bar">
            <div className={`dashboard-progress__fill ${isEngineActive ? 'dashboard-progress__fill--active' : ''}`} style={{ width: `${pct}%` }} />
          </div>
        </div>
        {/* File Inspector Panel */}
        {fileInspectorTab && (
          <div style={{
            marginTop: '16px', padding: '16px', borderRadius: 'var(--radius-md)',
            background: 'var(--c-hit-bg)', border: '1px solid var(--c-border)',
            maxHeight: '280px', overflowY: 'auto',
            animation: 'slide-in 150ms ease forwards',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
              <div style={{ display: 'flex', gap: '4px' }}>
                {['indexed', 'failed', 'skipped'].map(tab => {
                  const count = tab === 'indexed' ? indexed : tab === 'failed' ? errors : skipped;
                  if (count === 0 && tab !== 'indexed') return null;
                  return (
                    <button
                      key={tab}
                      onClick={() => {
                        if (tab === 'indexed' && fileInspectorTab !== 'indexed') {
                          setIndexedFilesLoading(true);
                          invoke<{ documents: IndexedFile[]; total: number }>('list_indexed_files')
                            .then(res => setIndexedFiles(res.documents || []))
                            .catch(err => console.error('[Dashboard] list_indexed_files failed:', err))
                            .finally(() => setIndexedFilesLoading(false));
                        }
                        setFileInspectorTab(tab as any);
                      }}
                      style={{
                        padding: '4px 10px', borderRadius: '6px', fontSize: '10px', fontWeight: 600,
                        textTransform: 'uppercase', letterSpacing: '0.04em', cursor: 'pointer',
                        border: '1px solid',
                        borderColor: fileInspectorTab === tab ? 'var(--c-accent)' : 'var(--c-border)',
                        background: fileInspectorTab === tab ? 'rgba(224,168,96,0.1)' : 'transparent',
                        color: fileInspectorTab === tab ? 'var(--c-accent)'
                          : tab === 'indexed' ? '#4ade80' : tab === 'failed' ? '#f87171' : '#facc15',
                        transition: 'all 200ms ease',
                      }}
                    >
                      {tab} ({count})
                    </button>
                  );
                })}
              </div>
              <button
                onClick={() => setFileInspectorTab(null)}
                style={{
                  background: 'none', border: 'none', cursor: 'pointer',
                  color: 'var(--c-text-muted)', padding: '2px',
                }}
                aria-label="Close file inspector"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <path d="M18 6L6 18M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Indexed tab */}
            {fileInspectorTab === 'indexed' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                {indexedFilesLoading ? (
                  <div style={{ textAlign: 'center', padding: '16px', color: 'var(--c-text-muted)', fontSize: '12px' }}>
                    <span className="spinner" style={{ width: '12px', height: '12px', marginRight: '8px' }} />
                    Loading indexed files…
                  </div>
                ) : indexedFiles.length === 0 ? (
                  <div style={{ textAlign: 'center', padding: '16px', color: 'var(--c-text-muted)', fontSize: '12px' }}>No indexed files found.</div>
                ) : indexedFiles.map((f, i) => (
                  <div key={i} style={{
                    padding: '6px 10px', borderRadius: 'var(--radius-sm)',
                    background: 'rgba(74,222,128,0.04)', border: '1px solid rgba(74,222,128,0.08)',
                    display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '8px',
                  }}>
                    <div style={{ minWidth: 0, flex: 1 }}>
                      <div style={{ fontSize: '11px', color: 'var(--c-text)', fontFamily: 'monospace', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {f.title || f.path.split('/').pop()}
                      </div>
                      <div style={{ fontSize: '9px', color: 'var(--c-text-muted)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{f.path}</div>
                    </div>
                    <span style={{ fontSize: '9px', color: 'var(--c-text-muted)', flexShrink: 0 }}>{f.mime_type}</span>
                  </div>
                ))}
              </div>
            )}

            {/* Failed tab */}
            {fileInspectorTab === 'failed' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                {failedFiles.length === 0 ? (
                  <div style={{ textAlign: 'center', padding: '16px', color: 'var(--c-text-muted)', fontSize: '12px' }}>No failed files.</div>
                ) : failedFiles.map((f, i) => (
                  <div key={i} style={{
                    padding: '6px 10px', borderRadius: 'var(--radius-sm)',
                    background: 'rgba(248,113,113,0.04)', border: '1px solid rgba(248,113,113,0.08)',
                    display: 'flex', flexDirection: 'column', gap: '2px',
                  }}>
                    <div style={{ display: 'flex', alignItems: 'flex-start', gap: '8px' }}>
                      <span style={{ fontSize: '9px', fontWeight: 600, color: '#f87171', padding: '2px 6px', background: 'rgba(248,113,113,0.1)', borderRadius: '4px', whiteSpace: 'nowrap', marginTop: '1px' }}>
                        {f.reason}
                      </span>
                      <span style={{ fontSize: '11px', color: 'var(--c-text)', fontFamily: 'monospace', wordBreak: 'break-all', marginTop: '1px' }}>
                        {f.path.split('/').pop() || f.path}
                      </span>
                    </div>
                    <span style={{ fontSize: '9px', color: 'var(--c-text-muted)', wordBreak: 'break-all', marginTop: '2px' }}>{f.path}</span>
                  </div>
                ))}
              </div>
            )}

            {/* Skipped tab */}
            {fileInspectorTab === 'skipped' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                {skippedFiles.length === 0 ? (
                  <div style={{ textAlign: 'center', padding: '16px', color: 'var(--c-text-muted)', fontSize: '12px' }}>No skipped files.</div>
                ) : skippedFiles.map((f, i) => (
                  <div key={i} style={{
                    padding: '6px 10px', borderRadius: 'var(--radius-sm)',
                    background: 'rgba(250,204,21,0.04)', border: '1px solid rgba(250,204,21,0.06)',
                    display: 'flex', flexDirection: 'column', gap: '2px',
                  }}>
                    <div style={{ display: 'flex', alignItems: 'flex-start', gap: '8px' }}>
                      <span style={{ fontSize: '9px', fontWeight: 600, color: '#facc15', padding: '2px 6px', background: 'rgba(250,204,21,0.1)', borderRadius: '4px', whiteSpace: 'nowrap', marginTop: '1px' }}>
                        {f.reason}
                      </span>
                      <span style={{ fontSize: '11px', color: 'var(--c-text)', fontFamily: 'monospace', wordBreak: 'break-all', marginTop: '1px' }}>
                        {f.path.split('/').pop() || f.path}
                      </span>
                    </div>
                    <span style={{ fontSize: '9px', color: 'var(--c-text-muted)', wordBreak: 'break-all', marginTop: '2px' }}>{f.path}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </header>

      {/* BENTO GRID */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '20px', position: 'relative', zIndex: 1 }}>
        
        {/* FOLDER MATRIX (spans 2 cols on wide) */}
        <section className={glassCard} style={{ padding: '24px', gridColumn: 'span 2', display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <h2 className={sectionTitle} style={{ margin: 0 }}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
              </svg>
              Monitored Folders
            </h2>
            <button className="managed-folder-add-btn" style={{ width: 'auto', margin: 0, padding: '6px 12px', fontSize: '11px' }} onClick={handleAddFolder} disabled={folderUpdating}>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path strokeLinecap="round" d="M12 4v16m8-8H4"/></svg>
              Add
            </button>
          </div>
          <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '4px' }}>
            {managedFolders.length === 0 ? (
              <div className="dashboard-empty"><span className="dashboard-empty__text">No folders registered. Add folders to start indexing.</span></div>
            ) : managedFolders.map((path) => {
              const pathLower = path.toLowerCase();
              const dirEntry = dirs.find(d => pathLower.includes(d.path.replace(/^~\//, '').toLowerCase()) || d.path.toLowerCase().includes(pathLower));
              const state = dirEntry ? dirEntry.state : 'completed';
              return (
                <div key={path} className="dir-list__item">
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '2px', minWidth: 0 }}>
                    <span className="dir-list__path" title={path}>{truncatePath(path)}</span>
                    <span className={`dir-badge dir-badge--${state}`} style={{ alignSelf: 'flex-start' }}>{state}</span>
                  </div>
                  <button className="dir-list__remove" onClick={() => handleRemoveFolder(path)} disabled={folderUpdating} title="Remove">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                    </svg>
                  </button>
                </div>
              );
            })}
          </div>
        </section>

        {/* THREAD VISUALIZER */}
        <section className={glassCard} style={{ padding: '24px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between', position: 'relative', overflow: 'hidden' }}>
          <h2 className={sectionTitle}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
            Live Threads
          </h2>
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative' }}>
            <div style={{
              width: '48px', height: '48px', borderRadius: '50%',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              background: 'var(--c-hit-bg)', border: '1px solid var(--c-border)',
              backdropFilter: 'blur(8px)', zIndex: 2,
              boxShadow: isEngineActive ? '0 0 20px rgba(224,168,96,0.2)' : 'none',
            }}>
              <div style={{
                width: '14px', height: '14px', borderRadius: '50%',
                background: paused ? '#71717a' : isEngineActive ? 'var(--c-accent)' : '#52525b',
                boxShadow: isEngineActive ? '0 0 15px rgba(224,168,96,0.8)' : 'none',
                animation: isEngineActive ? 'pulse 1.5s ease infinite' : 'none',
              }} />
            </div>
            {Array.from({ length: config.threads || 2 }).map((_, i, arr) => {
              const angle = (360 / arr.length) * i;
              const d = 42;
              const x = Math.cos((angle * Math.PI) / 180) * d;
              const y = Math.sin((angle * Math.PI) / 180) * d;
              return (
                <div key={i} style={{
                  position: 'absolute', transform: `translate(${x}px, ${y}px)`,
                  width: '22px', height: '22px', borderRadius: '50%',
                  background: 'var(--c-hit-bg)', border: `1px solid ${isEngineActive ? 'rgba(224,168,96,0.5)' : 'var(--c-border)'}`,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: '9px', fontWeight: 600, color: isEngineActive ? 'var(--c-accent)' : 'var(--c-text-muted)',
                  backdropFilter: 'blur(4px)', transition: 'all 0.5s ease',
                  animation: isEngineActive ? 'pulse 1.5s ease infinite' : 'none',
                }}>T{i+1}</div>
              );
            })}
          </div>
          <div style={{ textAlign: 'center', fontSize: '10px', color: 'var(--c-text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            {isEngineActive ? `${config.threads} thread workers active` : paused ? 'Suspended' : 'Idle'}
          </div>
        </section>

        {/* SYSTEM METRICS */}
        <section className={glassCard} style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <h2 className={sectionTitle}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2m0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 002 2h2a2 2 0 002-2z"/></svg>
            System Metrics
          </h2>
          {[
            { label: 'Database Size', value: formatBytes(dbSize) },
            { label: 'ONNX RAM', value: formatBytes(onnxMemory) },
          ].map(m => (
            <div key={m.label} style={{
              padding: '12px', borderRadius: 'var(--radius-md)',
              background: 'var(--c-hit-bg)', border: '1px solid var(--c-border)',
              display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            }}>
              <div style={{ display: 'flex', flexDirection: 'column' }}>
                <span style={{ fontSize: '10px', fontWeight: 600, color: 'var(--c-text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>{m.label}</span>
                <span style={{ fontSize: '16px', fontWeight: 700, fontFamily: 'monospace' }}>{m.value}</span>
              </div>
            </div>
          ))}
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'var(--c-text-muted)' }}>
            <span>Indexed Documents:</span>
            <span style={{ color: 'var(--c-accent)', fontWeight: 700, fontFamily: 'monospace' }}>{docCount.toLocaleString()}</span>
          </div>
        </section>

        {/* ENGINE TUNING — spans full width */}
        <section className={glassCard} style={{ padding: '24px', gridColumn: '1 / -1', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <h2 className={sectionTitle}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"/></svg>
            Engine Tuning (Failed: {failedFiles.length}, Skipped: {skippedFiles.length})
          </h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '24px' }}>
            {[
              { id: 'parallelism', label: 'Index Parallelism', value: config.indexParallelism, min: 1, max: 8, onChange: handleIndexParallelismChange },
              { id: 'threads', label: 'CPU Threads', value: config.threads, min: 1, max: 16, onChange: handleThreadsChange },
              { id: 'batch', label: 'Batch Size', value: config.batchSize, min: 1, max: 50, onChange: handleBatchSizeChange },
            ].map(s => (
              <div key={s.id} className="control-row">
                <label htmlFor={`${s.id}-slider`} className="control-label">
                  {s.label}
                  <span className="control-value">{s.value}</span>
                </label>
                <input id={`${s.id}-slider`} type="range" min={s.min} max={s.max} step={1} value={s.value} onChange={e => s.onChange(parseInt(e.target.value))} className="control-slider" />
                <div className="control-range-labels"><span>{s.min}</span><span>{s.max}</span></div>
              </div>
            ))}
          </div>
          <p className="control-hint">Changes are applied live — the engine picks up new values on its next batch.</p>
        </section>


      </div>

      <footer style={{ textAlign: 'center', fontSize: '10px', color: 'var(--c-score-text)', letterSpacing: '0.1em', textTransform: 'uppercase', marginTop: '8px', position: 'relative', zIndex: 1 }}>
        CRUMBS ENGINE • v0.1.0
      </footer>
    </div>
  );
}
