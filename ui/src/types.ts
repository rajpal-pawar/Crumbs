// types.ts — Crumbs IPC wire types mirroring crumbs-daemon's JSON responses.

/** Which search pipelines contributed to this hit. */
export interface HitSources {
  bm25:     boolean;
  vector:   boolean;
  fallback: boolean;
}

/** A single search result returned by the daemon. */
export interface SearchHit {
  doc_id:  number;
  path:    string;
  title:   string;
  snippet: string | null;
  score:   number;
  sources: HitSources;
}

/** Full payload returned by the `search` IPC command. */
export interface SearchResponse {
  hits:  SearchHit[];
  total: number;
}

/** Visual classification of a hit for badge rendering. */
export type MatchType = 'Strong' | 'Semantic' | 'Content' | 'Metadata';

/** Derive a human-readable match type from sources flags. */
export function classifyHit(sources: HitSources): MatchType {
  if (sources.bm25 && sources.vector) return 'Strong';
  if (sources.vector && !sources.bm25) return 'Semantic';
  if (sources.bm25  && !sources.vector) return 'Content';
  return 'Metadata';
}

/** Return the CSS class for a match-type badge. */
export function badgeClass(type: MatchType): string {
  switch (type) {
    case 'Strong':   return 'badge badge-strong';
    case 'Semantic': return 'badge badge-semantic';
    case 'Content':  return 'badge badge-content';
    case 'Metadata': return 'badge badge-meta';
  }
}
