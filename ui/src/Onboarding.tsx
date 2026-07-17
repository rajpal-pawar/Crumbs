// Onboarding.tsx — First-run setup flow
//
// Shown when is_onboarded is false. Prompts the user to select folders to
// index, then saves the config via IPC and kicks off the initial reindex.

import { useState, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';

// ---------------------------------------------------------------------------
// SVG Icons
// ---------------------------------------------------------------------------

const FolderPlusIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
    <line x1="12" y1="11" x2="12" y2="17" />
    <line x1="9" y1="14" x2="15" y2="14" />
  </svg>
);

const TrashIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="3 6 5 6 21 6" />
    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
  </svg>
);

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface OnboardingProps {
  onComplete: () => void;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function Onboarding({ onComplete }: OnboardingProps) {
  const [folders, setFolders] = useState<string[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSelectFolders = useCallback(async () => {
    try {
      const paths: string[] = await invoke('select_folders_dialog');
      if (paths && paths.length > 0) {
        // Merge with existing, avoiding duplicates
        setFolders(prev => {
          const combined = [...prev];
          for (const p of paths) {
            if (!combined.includes(p)) {
              combined.push(p);
            }
          }
          return combined;
        });
      }
    } catch (err) {
      console.error('[Crumbs] folder dialog error:', err);
      setError('Failed to open folder picker. Please try again.');
    }
  }, []);

  const handleRemoveFolder = useCallback((path: string) => {
    setFolders(prev => prev.filter(p => p !== path));
  }, []);

  const handleStart = useCallback(async () => {
    if (folders.length === 0) {
      setError('Please select at least one folder to index.');
      return;
    }
    setIsSubmitting(true);
    setError(null);

    try {
      await invoke('update_monitored_folders', {
        folders,
        isOnboarded: true,
      });
      onComplete();
    } catch (err) {
      console.error('[Crumbs] onboarding submission error:', err);
      alert(err);
      setError(`Failed to save configuration: ${err}`);
      setIsSubmitting(false);
    }
  }, [folders, onComplete]);

  const truncatePath = (path: string, maxLen = 55) => {
    if (path.length <= maxLen) return path;
    const half = Math.floor((maxLen - 3) / 2);
    return path.slice(0, half) + '…' + path.slice(-half);
  };

  return (
    <div className="onboarding-overlay">
      <div className="onboarding-panel">
        {/* Decorative background glow */}
        <div className="onboarding-glow" aria-hidden="true" />

        {/* Header */}
        <div className="onboarding-header">
          <img src="/logo-transparent.png" alt="Crumbs Logo" style={{ width: '220px', height: 'auto', objectFit: 'contain', marginBottom: '4px' }} />
          <h1 className="onboarding-title">Welcome to Crumbs</h1>
          <p className="onboarding-subtitle">
            Your intelligent, on-device file search engine.
            <br />
            Select the folders you'd like Crumbs to index.
          </p>
        </div>

        {/* Folder selection area */}
        <div className="onboarding-body">
          {folders.length > 0 && (
            <ul className="onboarding-folder-list">
              {folders.map((f) => (
                <li key={f} className="onboarding-folder-item">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="onboarding-folder-icon">
                    <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
                  </svg>
                  <span className="onboarding-folder-path" title={f}>
                    {truncatePath(f)}
                  </span>
                  <button
                    className="onboarding-folder-remove"
                    onClick={() => handleRemoveFolder(f)}
                    aria-label={`Remove ${f}`}
                    title="Remove folder"
                  >
                    <TrashIcon />
                  </button>
                </li>
              ))}
            </ul>
          )}

          <button
            className="onboarding-add-btn"
            onClick={handleSelectFolders}
            disabled={isSubmitting}
          >
            <FolderPlusIcon />
            <span>{folders.length === 0 ? 'Select Folders to Index' : 'Add More Folders'}</span>
          </button>

          {error && (
            <div className="onboarding-error" role="alert">
              {error}
            </div>
          )}
        </div>

        {/* Footer action */}
        <div className="onboarding-footer">
          <button
            className="onboarding-start-btn"
            onClick={handleStart}
            disabled={folders.length === 0 || isSubmitting}
          >
            {isSubmitting ? (
              <>
                <span className="spinner" aria-hidden="true" style={{ width: '14px', height: '14px', marginRight: '8px' }} />
                Starting…
              </>
            ) : (
              'Start Indexing'
            )}
          </button>
          <p className="onboarding-hint">
            You can always add or remove folders later in Settings.
          </p>
        </div>
      </div>
    </div>
  );
}
