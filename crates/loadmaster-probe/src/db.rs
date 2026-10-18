//! Eval results in SQLite (`~/.loadmaster/eval.db`).

use std::path::Path;

use rusqlite::{Connection, params};
use serde::Serialize;

use crate::eval::RowResult;

pub struct Db(Connection);

#[derive(Debug, Clone, Serialize)]
pub struct RunMeta {
    pub id: i64,
    pub created_at: String,
    pub url: String,
    pub model: String,
    pub runtime: String,
    pub runtime_version: Option<String>,
    pub prompts: String,
    pub prompts_blake3: String,
    pub n: i64,
}

/// One stored row, as read back.
#[derive(Debug, Clone, PartialEq)]
pub struct StoredRow {
    pub row_id: String,
    pub ok: bool,
    pub schema_valid: Option<bool>,
    pub latency_ms: u64,
    pub tokens_per_s: Option<f64>,
    pub response_blake3: String,
}

/// What a run scored. `accuracy` and `schema_*` depend only on the outputs, so a
/// deterministic runtime reproduces them exactly; timings are measurements and vary.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Summary {
    pub n: u64,
    pub ok: u64,
    pub schema_n: u64,
    pub schema_ok: u64,
    pub tokens_per_s_mean: Option<f64>,
    pub latency_p50_ms: u64,
    pub latency_p95_ms: u64,
}

impl Db {
    pub fn open(path: &Path) -> rusqlite::Result<Self> {
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let c = Connection::open(path)?;
        c.execute_batch(
            "CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, url TEXT NOT NULL,
                model TEXT NOT NULL, runtime TEXT NOT NULL, runtime_version TEXT,
                prompts TEXT NOT NULL, prompts_blake3 TEXT NOT NULL, probe_version TEXT NOT NULL,
                n INTEGER NOT NULL);
             CREATE TABLE IF NOT EXISTS results (
                run_id INTEGER NOT NULL REFERENCES runs(id), idx INTEGER NOT NULL, row_id TEXT NOT NULL,
                ok INTEGER NOT NULL, schema_valid INTEGER, tool_ok INTEGER, latency_ms INTEGER NOT NULL,
                completion_tokens INTEGER, tokens_per_s REAL, error TEXT, response_blake3 TEXT NOT NULL,
                PRIMARY KEY (run_id, idx));",
        )?;
        Ok(Db(c))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn insert(
        &mut self,
        created_at: &str,
        url: &str,
        model: &str,
        runtime: &str,
        runtime_version: Option<&str>,
        prompts: &str,
        prompts_blake3: &str,
        rows: &[RowResult],
    ) -> rusqlite::Result<i64> {
        let tx = self.0.transaction()?;
        tx.execute(
            "INSERT INTO runs (created_at, url, model, runtime, runtime_version, prompts, prompts_blake3, probe_version, n)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![created_at, url, model, runtime, runtime_version, prompts, prompts_blake3, crate::PROBE_VERSION, rows.len() as i64],
        )?;
        let id = tx.last_insert_rowid();
        for r in rows {
            tx.execute(
                "INSERT INTO results VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                params![
                    id,
                    r.idx as i64,
                    r.id,
                    r.ok,
                    r.schema_valid,
                    r.tool_ok,
                    r.latency_ms as i64,
                    r.completion_tokens.map(|t| t as i64),
                    r.tokens_per_s,
                    r.error,
                    r.response_blake3
                ],
            )?;
        }
        tx.commit()?;
        Ok(id)
    }

    pub fn run(&self, id: i64) -> rusqlite::Result<RunMeta> {
        self.0.query_row(
            "SELECT id, created_at, url, model, runtime, runtime_version, prompts, prompts_blake3, n FROM runs WHERE id = ?1",
            [id],
            |r| {
                Ok(RunMeta {
                    id: r.get(0)?,
                    created_at: r.get(1)?,
                    url: r.get(2)?,
                    model: r.get(3)?,
                    runtime: r.get(4)?,
                    runtime_version: r.get(5)?,
                    prompts: r.get(6)?,
                    prompts_blake3: r.get(7)?,
                    n: r.get(8)?,
                })
            },
        )
    }

    pub fn results(&self, id: i64) -> rusqlite::Result<Vec<StoredRow>> {
        let mut st = self.0.prepare(
            "SELECT row_id, ok, schema_valid, latency_ms, tokens_per_s, response_blake3 FROM results WHERE run_id = ?1 ORDER BY idx",
        )?;
        st.query_map([id], |r| {
            Ok(StoredRow {
                row_id: r.get(0)?,
                ok: r.get(1)?,
                schema_valid: r.get(2)?,
                latency_ms: r.get::<_, i64>(3)? as u64,
                tokens_per_s: r.get(4)?,
                response_blake3: r.get(5)?,
            })
        })?
        .collect()
    }

    pub fn summary(&self, id: i64) -> rusqlite::Result<Summary> {
        let rows = self.results(id)?;
        let mut lat: Vec<u64> = rows.iter().map(|r| r.latency_ms).collect();
        lat.sort_unstable();
        let pct = |p: f64| -> u64 {
            if lat.is_empty() {
                return 0;
            }
            let i = ((p * lat.len() as f64).ceil() as usize).clamp(1, lat.len()) - 1;
            lat[i]
        };
        let tps: Vec<f64> = rows.iter().filter_map(|r| r.tokens_per_s).collect();
        Ok(Summary {
            n: rows.len() as u64,
            ok: rows.iter().filter(|r| r.ok).count() as u64,
            schema_n: rows.iter().filter(|r| r.schema_valid.is_some()).count() as u64,
            schema_ok: rows.iter().filter(|r| r.schema_valid == Some(true)).count() as u64,
            tokens_per_s_mean: (!tps.is_empty())
                .then(|| tps.iter().sum::<f64>() / tps.len() as f64),
            latency_p50_ms: pct(0.50),
            latency_p95_ms: pct(0.95),
        })
    }
}
