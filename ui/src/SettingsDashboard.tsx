// SettingsDashboard.tsx — Settings & Status Dashboard (slide-out panel)
//
// Shows:
//   1. Directory Matrix — live per-directory indexing status from Tauri events.
//   2. Engine Controls — sliders for Batch Size and CPU Threads, debounced.

import { useState, useEffect, useRef, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

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

interface EngineConfig {
  batchSize: number;
  threads: number;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Middle-truncate a path for display. */
function truncatePath(path: string, maxLen = 40): string {
  if (path.length <= maxLen) return path;
  const half = Math.floor((maxLen - 3) / 2);
  return path.slice(0, half) + '…' + path.slice(-half);
}

/** Debounce utility — returns a wrapper that delays the call. */
function useDebouncedCallback<T extends (...args: any[]) => void>(
  fn: T,
  delayMs: number,
): T {
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const fnRef = useRef(fn);
  fnRef.current = fn;

  return useCallback(
    ((...args: any[]) => {
      if (timerRef.current) clearTimeout(timerRef.current);
      timerRef.current = setTimeout(() => fnRef.current(...args), delayMs);
    }) as unknown as T,
    [delayMs],
  );
}

// ---------------------------------------------------------------------------
// Status Badge Component
// ---------------------------------------------------------------------------

function StatusBadge({ state }: { state: DirEntry['state'] }) {
  const labels: Record<DirEntry['state'], string> = {
    queued: 'Queued',
    scanning: 'Scanning',
    indexing: 'Indexing',
    completed: 'Completed',
  };

  return <span className={`dir-badge dir-badge--${state}`}>{labels[state]}</span>;
}

// ---------------------------------------------------------------------------
// Animated Pulse Dot
// ---------------------------------------------------------------------------

function PulseDot({ active }: { active: boolean }) {
  if (!active) return null;
  return <span className="pulse-dot" aria-hidden="true" />;
}

// ---------------------------------------------------------------------------
// Main Dashboard Component
// ---------------------------------------------------------------------------

export interface SettingsDashboardProps {
  open: boolean;
  onClose: () => void;
}

export default function SettingsDashboard({ open, onClose }: SettingsDashboardProps) {
  const [dirs, setDirs] = useState<DirEntry[]>([]);
  const [indexed, setIndexed] = useState(0);
  const [total, setTotal] = useState(0);
  const [status, setStatus] = useState<string>('idle');
  const [config, setConfig] = useState<EngineConfig>({ batchSize: 5, threads: 2 });
  const [managedFolders, setManagedFolders] = useState<string[]>([]);
  const [folderUpdating, setFolderUpdating] = useState(false);
  const panelRef = useRef<HTMLDivElement>(null);

  // ── Load managed folders when panel opens ──
  useEffect(() => {
    if (!open) return;
    invoke<{ is_onboarded: boolean; watch_dirs: string[] }>('get_onboarding_status')
      .then((status) => {
        setManagedFolders(status.watch_dirs || []);
      })
      .catch((err) => console.error('[Crumbs] failed to load folders:', err));
  }, [open]);

  // ── Listen for progress events ──
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

  // ── Close on Escape ──
  useEffect(() => {
    if (!open) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        onClose();
      }
    };
    window.addEventListener('keydown', handleKey, true);
    return () => window.removeEventListener('keydown', handleKey, true);
  }, [open, onClose]);

  // ── Close on click outside ──
  useEffect(() => {
    if (!open) return;
    const handleClick = (e: MouseEvent) => {
      if (panelRef.current && !panelRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    // Delay the listener to prevent the opening click from immediately closing
    const timer = setTimeout(() => {
      document.addEventListener('mousedown', handleClick);
    }, 100);
    return () => {
      clearTimeout(timer);
      document.removeEventListener('mousedown', handleClick);
    };
  }, [open, onClose]);

  // ── Debounced config sender ──
  const sendConfig = useDebouncedCallback(
    (newConfig: EngineConfig) => {
      invoke('update_engine_config', {
        batchSize: newConfig.batchSize,
        threads: newConfig.threads,
      }).catch((err) => console.error('[Crumbs] config update failed:', err));
    },
    400,
  );

  const handleBatchSize = (val: number) => {
    const next = { ...config, batchSize: val };
    setConfig(next);
    sendConfig(next);
  };

  const handleThreads = (val: number) => {
    const next = { ...config, threads: val };
    setConfig(next);
    sendConfig(next);
  };

  // ── Managed Folders: Add ──
  const handleAddFolder = async () => {
    try {
      const paths: string[] = await invoke('select_folders_dialog');
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
      }
    } catch (err) {
      console.error('[Crumbs] add folder failed:', err);
      setFolderUpdating(false);
    }
  };

  // ── Managed Folders: Remove ──
  const handleRemoveFolder = async (path: string) => {
    const updated = managedFolders.filter(p => p !== path);
    setFolderUpdating(true);
    try {
      await invoke('update_monitored_folders', { folders: updated, isOnboarded: true });
      setManagedFolders(updated);
    } catch (err) {
      console.error('[Crumbs] remove folder failed:', err);
    }
    setFolderUpdating(false);
  };

  if (!open) return null;

  const isActive = status === 'indexing' || status === 'scanning';
  const pct = total > 0 ? Math.round((indexed / total) * 100) : 0;

  return (
    <div className="dashboard-overlay" aria-modal="true" role="dialog" aria-label="Settings Dashboard">
      <div ref={panelRef} className={`dashboard-panel ${open ? 'dashboard-panel--open' : ''}`}>
        {/* Header */}
        <div className="dashboard-header">
          <div className="dashboard-header__left">
            <svg className="dashboard-header__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="3" />
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
            </svg>
            <h2 className="dashboard-title">Engine Dashboard</h2>
          </div>
          <button className="dashboard-close" onClick={onClose} aria-label="Close dashboard">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Progress Bar */}
        <div className="dashboard-progress">
          <div className="dashboard-progress__bar">
            <div
              className={`dashboard-progress__fill ${isActive ? 'dashboard-progress__fill--active' : ''}`}
              style={{ width: `${pct}%` }}
            />
          </div>
          <div className="dashboard-progress__label">
            <PulseDot active={isActive} />
            <span>{isActive ? `Indexing ${indexed.toLocaleString()} / ${total.toLocaleString()} files…` : total > 0 ? `${total.toLocaleString()} files indexed` : 'Idle'}</span>
            {total > 0 && <span className="dashboard-progress__pct">{pct}%</span>}
          </div>
        </div>

        {/* Managed Folders — CRUD for watch_dirs */}
        <section className="dashboard-section">
          <h3 className="dashboard-section__title">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
            </svg>
            Managed Folders
          </h3>

          {managedFolders.length === 0 ? (
            <div className="dashboard-empty">
              <span className="dashboard-empty__text">No folders configured. Add folders to start indexing.</span>
            </div>
          ) : (
            <ul className="dir-list">
              {managedFolders.map((path) => (
                <li key={path} className="dir-list__item">
                  <span className="dir-list__path" title={path}>
                    {truncatePath(path)}
                  </span>
                  <button
                    className="dir-list__remove"
                    onClick={() => handleRemoveFolder(path)}
                    disabled={folderUpdating}
                    aria-label={`Remove ${path}`}
                    title="Remove folder"
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="3 6 5 6 21 6" />
                      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                    </svg>
                  </button>
                </li>
              ))}
            </ul>
          )}

          <button
            className="managed-folder-add-btn"
            onClick={handleAddFolder}
            disabled={folderUpdating}
          >
            {folderUpdating ? (
              <>
                <span className="spinner" style={{ width: '12px', height: '12px' }} />
                Updating…
              </>
            ) : (
              <>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
                  <line x1="12" y1="11" x2="12" y2="17" />
                  <line x1="9" y1="14" x2="15" y2="14" />
                </svg>
                Add Folder
              </>
            )}
          </button>
        </section>

        {/* Indexing Status Matrix */}
        {dirs.length > 0 && (
          <section className="dashboard-section">
            <h3 className="dashboard-section__title">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="4" y1="21" x2="4" y2="14" />
                <line x1="4" y1="10" x2="4" y2="3" />
                <line x1="12" y1="21" x2="12" y2="12" />
                <line x1="12" y1="8" x2="12" y2="3" />
                <line x1="20" y1="21" x2="20" y2="16" />
                <line x1="20" y1="12" x2="20" y2="3" />
                <line x1="1" y1="14" x2="7" y2="14" />
                <line x1="9" y1="8" x2="15" y2="8" />
                <line x1="17" y1="16" x2="23" y2="16" />
              </svg>
              Indexing Status
            </h3>
            <ul className="dir-list">
              {dirs.map((d, i) => (
                <li key={i} className="dir-list__item">
                  <span className="dir-list__path" title={d.path}>
                    {truncatePath(d.path)}
                  </span>
                  <StatusBadge state={d.state} />
                </li>
              ))}
            </ul>
          </section>
        )}

        {/* Engine Controls */}
        <section className="dashboard-section">
          <h3 className="dashboard-section__title">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="4" y1="21" x2="4" y2="14" />
              <line x1="4" y1="10" x2="4" y2="3" />
              <line x1="12" y1="21" x2="12" y2="12" />
              <line x1="12" y1="8" x2="12" y2="3" />
              <line x1="20" y1="21" x2="20" y2="16" />
              <line x1="20" y1="12" x2="20" y2="3" />
              <line x1="1" y1="14" x2="7" y2="14" />
              <line x1="9" y1="8" x2="15" y2="8" />
              <line x1="17" y1="16" x2="23" y2="16" />
            </svg>
            Engine Tuning
          </h3>

          <div className="control-group">
            <div className="control-row">
              <label htmlFor="batch-size-slider" className="control-label">
                Batch Size
                <span className="control-value">{config.batchSize}</span>
              </label>
              <input
                id="batch-size-slider"
                type="range"
                min={1}
                max={50}
                step={1}
                value={config.batchSize}
                onChange={(e) => handleBatchSize(Number(e.target.value))}
                className="control-slider"
              />
              <div className="control-range-labels">
                <span>1</span>
                <span>50</span>
              </div>
            </div>

            <div className="control-row">
              <label htmlFor="threads-slider" className="control-label">
                CPU Threads
                <span className="control-value">{config.threads}</span>
              </label>
              <input
                id="threads-slider"
                type="range"
                min={1}
                max={16}
                step={1}
                value={config.threads}
                onChange={(e) => handleThreads(Number(e.target.value))}
                className="control-slider"
              />
              <div className="control-range-labels">
                <span>1</span>
                <span>16</span>
              </div>
            </div>
          </div>

          <p className="control-hint">
            Changes are applied live — the engine picks up new values on its next batch iteration.
          </p>
        </section>

        {/* Footer */}
        <div className="dashboard-footer">
          <span className="dashboard-footer__version">Crumbs Engine v0.1.0</span>
        </div>
      </div>
    </div>
  );
}
