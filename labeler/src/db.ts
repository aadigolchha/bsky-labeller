/**
 * Database Manager for Bluesky Labeler
 * 
 * Handles persistent storage for:
 * - Jetstream cursor position (for resume after restart)
 * - Label history (for auditing and debugging)
 */

import Database from 'better-sqlite3';
import path from "path";

// Use /data volume in production for persistence across deploys
const dbPath = process.env.NODE_ENV === 'production' 
  ? '/data/labeler.db' 
  : 'labeler.db';

console.log(`📁 Database path: ${dbPath}`);

const db = new Database(dbPath);

// Initialize schema
db.exec(`
  -- State table for cursor and other key-value pairs
  CREATE TABLE IF NOT EXISTS state (
    key TEXT PRIMARY KEY,
    value INTEGER
  );

  -- Label history with extended metadata
  CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uri TEXT NOT NULL,
    cid TEXT,
    val TEXT NOT NULL,
    text TEXT,
    confidence REAL,
    theme TEXT,
    cluster_id INTEGER,
    indexedAt DATETIME DEFAULT CURRENT_TIMESTAMP
  );

  -- Index for faster lookups
  CREATE INDEX IF NOT EXISTS idx_history_val ON history(val);
  CREATE INDEX IF NOT EXISTS idx_history_indexed ON history(indexedAt);
`);

// Prepared statements for performance
const getCursorStmt = db.prepare('SELECT value FROM state WHERE key = ?');
const setCursorStmt = db.prepare('INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)');
const logLabelStmt = db.prepare(
  'INSERT INTO history (uri, cid, val, text, confidence, theme, cluster_id) VALUES (?, ?, ?, ?, ?, ?, ?)'
);
const getRecentLabelsStmt = db.prepare('SELECT * FROM history ORDER BY id DESC LIMIT ?');
const getLabelStatsStmt = db.prepare(`
  SELECT 
    val,
    COUNT(*) as count,
    AVG(confidence) as avg_confidence,
    MIN(indexedAt) as first_seen,
    MAX(indexedAt) as last_seen
  FROM history 
  GROUP BY val 
  ORDER BY count DESC
`);

/**
 * Get the current Jetstream cursor position.
 */
export const getCursor = (): number | undefined => {
  const row = getCursorStmt.get('cursor') as { value: number } | undefined;
  return row?.value;
};

/**
 * Save the Jetstream cursor position.
 */
export const setCursor = (cursor: number): void => {
  setCursorStmt.run('cursor', cursor);
};

/**
 * Log a label application with full metadata.
 */
export const logLabel = (
  uri: string,
  cid: string | undefined,
  val: string,
  text: string,
  confidence?: number,
  theme?: string,
  clusterId?: number
): void => {
  logLabelStmt.run(
    uri,
    cid ?? null,
    val,
    text?.slice(0, 500),  // Truncate text to save space
    confidence ?? null,
    theme ?? null,
    clusterId ?? null
  );
};

/**
 * Get recent label history.
 */
export const getRecentLabels = (limit = 10): any[] => {
  return getRecentLabelsStmt.all(limit);
};

/**
 * Get label statistics grouped by label type.
 */
export const getLabelStats = (): any[] => {
  return getLabelStatsStmt.all();
};

/**
 * Get count of labels by type for monitoring.
 */
export const getLabelCounts = (): Record<string, number> => {
  const stats = getLabelStats();
  return stats.reduce((acc, row) => {
    acc[row.val] = row.count;
    return acc;
  }, {} as Record<string, number>);
};

export default db;
