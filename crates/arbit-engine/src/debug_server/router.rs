use std::sync::{Arc, RwLock};

use axum::Router;

use crate::ProcessingEngine;

/// Build the Axum router used by the debug server.
pub fn build_router(_engine: Arc<RwLock<ProcessingEngine>>) -> Router {
    Router::new()
}
