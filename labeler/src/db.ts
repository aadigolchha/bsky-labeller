import Database from 'better-sqlite3';

const db = new Database('labeler.db');

// Initialize tables
db.exec(`
  CREATE TABLE IF NOT EXISTS state (
    key TEXT PRIMARY KEY,
    value INTEGER
  );

  CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uri TEXT NOT NULL,
    cid TEXT,
    val TEXT NOT NULL,
    text TEXT,
    indexedAt DATETIME DEFAULT CURRENT_TIMESTAMP
  );
`);

export const getCursor = (): number | undefined => {
  const row = db.prepare('SELECT value FROM state WHERE key = ?').get('cursor') as { value: number } | undefined;
  return row?.value;
};

export const setCursor = (cursor: number) => {
  db.prepare('INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)').run('cursor', cursor);
};

// UPDATED: Now accepts 'text' argument
export const logLabel = (uri: string, cid: string, val: string, text: string) => {
  db.prepare('INSERT INTO history (uri, cid, val, text) VALUES (?, ?, ?, ?)').run(uri, cid, val, text);
};

export const getRecentLabels = (limit = 10) => {
  return db.prepare('SELECT * FROM history ORDER BY id DESC LIMIT ?').all(limit);
};