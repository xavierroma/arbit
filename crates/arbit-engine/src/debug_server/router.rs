use std::sync::{Arc, RwLock};

use axum::{extract::State, response::IntoResponse, routing::get, Json, Router};

use crate::SlamEngine;

#[derive(Clone)]
struct AppState {
    engine: Arc<RwLock<SlamEngine>>,
}

pub fn build_router(engine: Arc<RwLock<SlamEngine>>) -> Router {
    let state = AppState { engine };
    Router::new()
        .route("/healthz", get(healthz))
        .route("/snapshot", get(snapshot))
        .with_state(state)
}

async fn healthz() -> impl IntoResponse {
    "ok"
}

async fn snapshot(State(state): State<AppState>) -> impl IntoResponse {
    let snapshot = state
        .engine
        .read()
        .expect("engine rwlock poisoned")
        .snapshot();
    Json(snapshot)
}
