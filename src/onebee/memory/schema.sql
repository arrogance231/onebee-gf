-- onebee memory schema
-- data/stores/*.db are versioned artifacts (data-versioning rule).
PRAGMA user_version = 1;

CREATE TABLE IF NOT EXISTS memory (
  id              TEXT PRIMARY KEY,
  tier            TEXT NOT NULL CHECK (tier IN ('short_term','episodic','semantic')),
  content         TEXT NOT NULL,
  content_struct  TEXT,
  embedding_id    INTEGER,
  created_at      INTEGER NOT NULL,
  event_time      INTEGER,
  valid_from      INTEGER,
  valid_to        INTEGER,
  last_accessed   INTEGER,
  access_count    INTEGER DEFAULT 0,
  importance      REAL NOT NULL,
  confidence      REAL NOT NULL,
  decay_rate      REAL NOT NULL,
  provenance      TEXT NOT NULL,
  emotion_valence REAL,
  emotion_arousal REAL,
  entities        TEXT,
  topics          TEXT,
  supersedes      TEXT,
  superseded_by   TEXT,
  contradiction_group TEXT,
  status          TEXT DEFAULT 'active' CHECK (status IN ('active','superseded','retracted','archived','quarantined')),
  redaction       INTEGER DEFAULT 0,
  schema_version  INTEGER NOT NULL DEFAULT 1
);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
  content, entities, topics,
  content='memory', content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS memory_ai AFTER INSERT ON memory BEGIN
  INSERT INTO memory_fts(rowid, content, entities, topics)
  VALUES (new.rowid, new.content, new.entities, new.topics);
END;

CREATE TRIGGER IF NOT EXISTS memory_ad AFTER DELETE ON memory BEGIN
  INSERT INTO memory_fts(memory_fts, rowid, content, entities, topics)
  VALUES ('delete', old.rowid, old.content, old.entities, old.topics);
END;

CREATE TRIGGER IF NOT EXISTS memory_au AFTER UPDATE ON memory BEGIN
  INSERT INTO memory_fts(memory_fts, rowid, content, entities, topics)
  VALUES ('delete', old.rowid, old.content, old.entities, old.topics);
  INSERT INTO memory_fts(rowid, content, entities, topics)
  VALUES (new.rowid, new.content, new.entities, new.topics);
END;

CREATE TABLE IF NOT EXISTS memory_edge (
  src TEXT,
  dst TEXT,
  kind TEXT,
  weight REAL
);

CREATE TABLE IF NOT EXISTS entity (
  id TEXT PRIMARY KEY,
  name TEXT,
  kind TEXT,
  aliases TEXT,
  salience REAL
);

CREATE TABLE IF NOT EXISTS session (
  id TEXT PRIMARY KEY,
  started INTEGER,
  ended INTEGER,
  summary TEXT,
  turn_count INTEGER,
  affect_summary TEXT
);

CREATE TABLE IF NOT EXISTS turn (
  turn_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('user','assistant')),
  text TEXT NOT NULL,
  ts INTEGER NOT NULL,
  tokens INTEGER,
  affect TEXT,
  entities TEXT,
  FOREIGN KEY(session_id) REFERENCES session(id)
);

CREATE INDEX IF NOT EXISTS idx_memory_tier ON memory(tier);
CREATE INDEX IF NOT EXISTS idx_memory_event_time ON memory(event_time);
CREATE INDEX IF NOT EXISTS idx_memory_status ON memory(status);
CREATE INDEX IF NOT EXISTS idx_turn_session ON turn(session_id, ts);
