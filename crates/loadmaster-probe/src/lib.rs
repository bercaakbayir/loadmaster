//! Tool-call and structured-output conformance probe, eval runner.
//!
//! [`run::probe`] runs declarative [`fixture`]s against any OpenAI-compatible endpoint and
//! classifies each failure; [`eval`] runs a user's prompts and stores results in SQLite.

pub mod check;
pub mod client;
pub mod compare;
pub mod db;
pub mod detect;
pub mod embed;
pub mod eval;
pub mod fake;
pub mod fixture;
pub mod known;
pub mod run;
pub mod time;

pub use client::Client;
pub use fixture::Fixture;
pub use run::{Options, Outcome, PROBE_VERSION, Report};
