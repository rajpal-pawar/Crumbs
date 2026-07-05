import { useState, useEffect, useRef, useCallback, KeyboardEvent } from 'react';
import './index.css';
import { useSearch } from './useSearch';
import { classifyHit, badgeClass, type SearchHit } from './types';
import SettingsDashboard from './SettingsDashboard';
import Onboarding from './Onboarding';
import { invoke } from '@tauri-apps/api/core';

// ---------------------------------------------------------------------------
// SVG icons (inline — no icon library dependency)
// ---------------------------------------------------------------------------

const SearchIcon = () => (
  <svg className="search-icon" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.8">
    <circle cx="8.5" cy="8.5" r="5.5" />
    <path d="M14.5 14.5 L18 18" strokeLinecap="round" />
  </svg>
);

const DocumentIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="file-icon">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
    <polyline points="14 2 14 8 20 8"></polyline>
    <line x1="16" y1="13" x2="8" y2="13"></line>
    <line x1="16" y1="17" x2="8" y2="17"></line>
    <polyline points="10 9 9 9 8 9"></polyline>
  </svg>
);

const ImageIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="file-icon">
    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
    <circle cx="8.5" cy="8.5" r="1.5"></circle>
    <polyline points="21 15 16 10 5 21"></polyline>
  </svg>
);

const CodeIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="file-icon">
    <polyline points="16 18 22 12 16 6"></polyline>
    <polyline points="8 6 2 12 8 18"></polyline>
  </svg>
);

function FileIcon({ filename }: { filename: string }) {
  const ext = filename.split('.').pop()?.toLowerCase();
  if (['png', 'jpg', 'jpeg', 'gif', 'webp'].includes(ext || '')) {
    return <ImageIcon />;
  }
  if (['txt', 'md', 'json', 'yaml', 'xml', 'csv', 'js', 'ts', 'jsx', 'tsx', 'py', 'rs', 'go', 'html', 'css'].includes(ext || '')) {
    return <CodeIcon />;
  }
  return <DocumentIcon />;
}

function middleTruncate(path: string, maxLength: number = 65) {
  if (path.length <= maxLength) return path;
  const half = Math.floor((maxLength - 3) / 2);
  return path.slice(0, half) + '...' + path.slice(-half);
}

// ---------------------------------------------------------------------------
// Hit row component
// ---------------------------------------------------------------------------

function HitRow({ hit, index, selected }: { hit: SearchHit; index: number; selected?: boolean }) {
  const matchType = classifyHit(hit.sources);
  const badge     = badgeClass(matchType);
  const ref       = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (selected && ref.current) {
      ref.current.scrollIntoView({ block: 'nearest' });
    }
  }, [selected]);

  // Show only the file name portion for readability; full path on hover.
  const filename = hit.title || hit.path.split(/[\\/]/).pop() || hit.path;
  const parentPath = hit.path.substring(0, Math.max(hit.path.lastIndexOf('/'), hit.path.lastIndexOf('\\'))) || hit.path;
  const dirPath  = middleTruncate(parentPath, 65);

  // Clicking opens the file via the shell.
  const handleOpen = useCallback(() => {
    console.info('[Crumbs] open:', hit.path);
    import('@tauri-apps/api/core').then(({ invoke }) => {
      invoke('open_file', { path: hit.path }).catch(console.error);
    });
  }, [hit.path]);

  const handleKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleOpen();
    }
  };

  return (
    <div
      ref={ref}
      id={`hit-${index}`}
      role="option"
      aria-selected={selected}
      tabIndex={0}
      className={`hit-row ${selected ? 'selected' : ''}`}
      onClick={handleOpen}
      onKeyDown={handleKeyDown}
      title={dirPath}
    >
      <div className="hit-icon-title">
        <FileIcon filename={filename} />
        <span className="hit-title">{filename}</span>
      </div>
      <span className="hit-path">{dirPath}</span>

      {hit.snippet && (
        <span
          className="hit-snippet"
          dangerouslySetInnerHTML={{ __html: hit.snippet }}
        />
      )}

      <div className="hit-meta">
        <span className={badge}>{matchType}</span>
        <span className="hit-score">{hit.score.toFixed(4)}</span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main App
// ---------------------------------------------------------------------------

export default function App() {
  const [query, setQuery]   = useState('');
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const [indexedCount, setIndexedCount]   = useState(0);
  const [totalCount, setTotalCount]       = useState(0);
  const inputRef            = useRef<HTMLInputElement>(null);
  const searchState         = useSearch(query);
  const [dashboardOpen, setDashboardOpen] = useState(false);
  const [isOnboarded, setIsOnboarded] = useState<boolean | null>(null); // null = loading
  const [onboardingChecked, setOnboardingChecked] = useState(false);

  // Check onboarding status on mount
  useEffect(() => {
    let cancelled = false;
    const checkOnboarding = async () => {
      try {
        const status = await invoke<{ is_onboarded: boolean; watch_dirs: string[] }>('get_onboarding_status');
        if (!cancelled) {
          setIsOnboarded(status.is_onboarded);
          setOnboardingChecked(true);
        }
      } catch (err) {
        console.error('[Crumbs] onboarding check failed:', err);
        // If the daemon isn't ready yet, retry after a brief delay
        if (!cancelled) {
          setTimeout(checkOnboarding, 1500);
        }
      }
    };
    // Small delay to let the daemon sidecar boot
    setTimeout(checkOnboarding, 800);
    return () => { cancelled = true; };
  }, []);

  const handleOnboardingComplete = useCallback(() => {
    setIsOnboarded(true);
  }, []);

  const showResults =
    searchState.status === 'results' ||
    searchState.status === 'loading' ||
    searchState.status === 'error';

  const hits = searchState.status === 'results' ? searchState.hits : [];
  const total = searchState.status === 'results' ? searchState.total : 0;

  useEffect(() => {
    setSelectedIndex(-1);
  }, [hits]);

  useEffect(() => {
    let unlistenProgress: (() => void) | undefined;
    import('@tauri-apps/api/event').then(({ listen }) => {
      listen<any>('crumbs://index-progress', (event) => {
        const payload = event.payload;
        if (!payload) return;
        const { indexed, total } = payload;
        if (indexed !== undefined && total !== undefined) {
          setIndexedCount(indexed);
          setTotalCount(total);
        }
      }).then(un => {
        unlistenProgress = un;
      });
    });
    return () => {
      if (unlistenProgress) unlistenProgress();
    };
  }, []);

  // -------------------------------------------------------------------------
  // Focus the input whenever the Tauri window gains focus.
  // The global shortcut (Ctrl+Space) is registered in src-tauri/src/lib.rs
  // and shows/focuses the window; this effect picks up from there.
  // -------------------------------------------------------------------------
  useEffect(() => {
    let unlistenTauriFocus: (() => void) | undefined;

    const focusInput = () => {
      setTimeout(() => {
        if (inputRef.current) {
          inputRef.current.focus();
          inputRef.current.select(); // Highlight all text
        }
      }, 50);
    };

    // Standard web focus event
    window.addEventListener('focus', focusInput);

    // Native Tauri focus event (fires when window.show() is called)
    import('@tauri-apps/api/window').then(({ getCurrentWindow }) => {
      getCurrentWindow().listen('tauri://focus', focusInput).then(un => {
        unlistenTauriFocus = un;
      });
    });

    // Focus immediately on mount
    focusInput();

    return () => {
      window.removeEventListener('focus', focusInput);
      if (unlistenTauriFocus) unlistenTauriFocus();
    };
  }, []);

  // -------------------------------------------------------------------------
  // Keyboard: Escape clears query / hides window. Arrows navigate.
  // -------------------------------------------------------------------------
  useEffect(() => {
    const handleGlobal = (e: globalThis.KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (query) {
          setQuery('');
        } else {
          inputRef.current?.blur();
        }
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedIndex(prev => (prev < hits.length - 1 ? prev + 1 : prev));
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedIndex(prev => (prev > 0 ? prev - 1 : prev > -1 ? -1 : prev));
      } else if (e.key === 'Enter') {
        // If they haven't explicitly navigated down but there are hits, default to the first hit.
        const targetIdx = selectedIndex >= 0 ? selectedIndex : (hits.length > 0 ? 0 : -1);
        if (targetIdx >= 0 && targetIdx < hits.length) {
          e.preventDefault();
          console.info('[Crumbs] open:', hits[targetIdx].path);
          import('@tauri-apps/api/core').then(({ invoke }) => {
            invoke('open_file', { path: hits[targetIdx].path }).catch(console.error);
          });
        }
      }
    };
    window.addEventListener('keydown', handleGlobal);
    return () => window.removeEventListener('keydown', handleGlobal);
  }, [query, hits, selectedIndex]);

  // ── Onboarding gate ──
  // Show loading splash while checking onboarding status
  if (!onboardingChecked || isOnboarded === null) {
    return (
      <div className="onboarding-overlay">
        <div className="onboarding-loading">
          <span className="spinner" style={{ width: '24px', height: '24px' }} />
          <span style={{ color: 'var(--c-text-muted)', marginTop: '12px', fontSize: '13px' }}>Connecting to Crumbs engine…</span>
        </div>
      </div>
    );
  }

  // Show onboarding flow if not yet onboarded
  if (!isOnboarded) {
    return <Onboarding onComplete={handleOnboardingComplete} />;
  }

  return (
    <div className="crumbs-shell" role="combobox" aria-haspopup="listbox" aria-expanded={showResults}>
      {/* ------------------------------------------------------------------ */}
      {/* Search bar                                                          */}
      {/* ------------------------------------------------------------------ */}
      <div className="search-bar">
        <SearchIcon />

        <input
          id="crumbs-search"
          ref={inputRef}
          className="search-input"
          type="search"
          autoComplete="off"
          autoCorrect="off"
          spellCheck={false}
          placeholder="Search your files…"
          aria-label="Search your files"
          aria-controls="crumbs-results"
          aria-autocomplete="list"
          value={query}
          onChange={e => setQuery(e.target.value)}
        />

        {!query && (
          <kbd className="kbd" aria-label="Press Ctrl+Space to open">
            <span>Ctrl</span><span>+</span><span>Space</span>
          </kbd>
        )}

        {searchState.status === 'loading' && (
          <span className="spinner" role="status" aria-label="Searching…" />
        )}

        <button
          id="settings-gear"
          className="gear-button"
          onClick={() => setDashboardOpen(true)}
          aria-label="Open settings dashboard"
          title="Engine Settings"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="3" />
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
          </svg>
        </button>
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Results panel                                                       */}
      {/* ------------------------------------------------------------------ */}
      {(showResults || !query) && (
        <div id="crumbs-results" className="results-panel" role="listbox" aria-label="Search results">
          {showResults && (
            <ul className="results-list" aria-live="polite" aria-atomic="false">
              {searchState.status === 'loading' && hits.length === 0 && (
                <li className="status-row">
                  <span className="spinner" aria-hidden="true" />
                  Searching…
                </li>
              )}

              {searchState.status === 'error' && (
                <li className="status-row" role="alert">
                  {searchState.message}
                </li>
              )}

              {searchState.status === 'results' && hits.length === 0 && (
                <li className="status-row">
                  No results for <strong style={{ color: 'var(--c-text)' }}>"{query}"</strong>
                </li>
              )}

              {hits.map((hit, i) => (
                <li key={hit.doc_id} role="presentation">
                  <HitRow hit={hit} index={i} selected={i === selectedIndex} />
                </li>
              ))}
            </ul>
          )}

          {/* Footer */}
          <div className="results-footer">
            <span className="daemon-status">
              {indexedCount < totalCount ? (
                <>
                  <span className="spinner" aria-hidden="true" style={{ width: '12px', height: '12px', display: 'inline-block', marginRight: '6px', verticalAlign: 'middle' }} />
                  Indexing: {indexedCount} / {totalCount} files...
                </>
              ) : totalCount > 0 ? (
                `System Ready • ${totalCount} files indexed.`
              ) : (
                'System Ready'
              )}
            </span>
            <span>
              <kbd className="kbd">↑↓</kbd> navigate &nbsp;
              <kbd className="kbd">↵</kbd> open &nbsp;
              <kbd className="kbd">Esc</kbd> dismiss
            </span>
          </div>
        </div>
      )}

      {/* Settings Dashboard */}
      <SettingsDashboard open={dashboardOpen} onClose={() => setDashboardOpen(false)} />
    </div>
  );
}
