// useSearch.ts — Debounced Tauri IPC search hook.

import { useState, useEffect, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import type { SearchHit, SearchResponse } from './types';

export type SearchState =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'results'; hits: SearchHit[]; total: number }
  | { status: 'error'; message: string };

/**
 * Sends a search query to the Crumbs daemon via Tauri IPC and returns
 * the current search state.
 *
 * The query is debounced to avoid hammering the daemon on every keystroke.
 * In-flight requests are cancelled (via AbortController pattern using a
 * generation counter) so stale responses never overwrite fresher ones.
 */
export function useSearch(query: string): SearchState {
  const [state, setState] = useState<SearchState>({ status: 'idle' });
  // Generation counter — incremented on every new search; stale callbacks
  // check if their generation is still current before calling setState.
  const generation = useRef(0);

  useEffect(() => {
    const trimmed = query.trim();

    if (!trimmed) {
      return;
    }


    // Bump generation so any in-flight request from a previous query
    // will see a stale generation and discard its result.
    const currentGen = ++generation.current;

    setState({ status: 'loading' });

    // If the user typed a space, they likely finished a word, so we search quickly.
    // Otherwise, wait longer (1000ms) to let them finish typing the word.
    const isWordComplete = query.endsWith(' ');
    const delayMs = isWordComplete ? 50 : 1000;

    const timer = setTimeout(async () => {
      try {
        const resp = await invoke<SearchResponse>('search', {
          query: trimmed,
          limit: 10,
        });

        // Only update state if this is still the latest query.
        if (generation.current === currentGen) {
          setState({ status: 'results', hits: resp.hits, total: resp.total });
        }
      } catch (err) {
        if (generation.current === currentGen) {
          setState({
            status: 'error',
            message: err instanceof Error ? err.message : String(err),
          });
        }
      }
    }, delayMs);

    return () => clearTimeout(timer);
  }, [query]);

  if (!query.trim()) {
    return { status: 'idle' };
  }

  return state;
}
