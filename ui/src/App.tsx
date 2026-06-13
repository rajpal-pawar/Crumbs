import { useState, useEffect, useRef, useCallback, KeyboardEvent } from 'react';
import './index.css';
import { useSearch } from './useSearch';
import { classifyHit, badgeClass, type SearchHit } from './types';

// ---------------------------------------------------------------------------
// SVG icons (inline — no icon library dependency)
// ---------------------------------------------------------------------------

const SearchIcon = () => (
  <svg className="search-icon" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.8">
    <circle cx="8.5" cy="8.5" r="5.5" />
    <path d="M14.5 14.5 L18 18" strokeLinecap="round" />
  </svg>
);

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
  const dirPath  = hit.path;

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
      <span className="hit-title">{filename}</span>
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
  const inputRef            = useRef<HTMLInputElement>(null);
  const searchState         = useSearch(query);

  const showResults =
    searchState.status === 'results' ||
    searchState.status === 'loading' ||
    searchState.status === 'error';

  const hits = searchState.status === 'results' ? searchState.hits : [];
  const total = searchState.status === 'results' ? searchState.total : 0;

  useEffect(() => {
    setSelectedIndex(-1);
  }, [hits]);

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
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Results panel                                                       */}
      {/* ------------------------------------------------------------------ */}
      {showResults && (
        <div id="crumbs-results" className="results-panel" role="listbox" aria-label="Search results">
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

          {/* Footer */}
          {hits.length > 0 && (
            <div className="results-footer">
              <span>{total} result{total !== 1 ? 's' : ''}</span>
              <span>
                <kbd className="kbd">↑↓</kbd> navigate &nbsp;
                <kbd className="kbd">↵</kbd> open &nbsp;
                <kbd className="kbd">Esc</kbd> dismiss
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
