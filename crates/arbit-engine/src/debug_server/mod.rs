//! Debug server runtime scaffolding.
//!
//! This module wires up the background Tokio runtime that hosts the HTTP/
//! WebSocket debug surface. Higher level routing and protocol concerns are
//! built in subsequent phases; here we focus on lifecycle management and
//! graceful shutdown semantics.

mod router;

use std::fmt;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::{Arc, RwLock};

use axum::Router;
use tokio::runtime::{Builder as RuntimeBuilder, Runtime};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use crate::ProcessingEngine;

/// Configuration options controlling the debug server runtime.
#[derive(Debug, Clone)]
pub struct DebugServerOpts {
    /// TCP port to bind. Use `0` for an ephemeral port.
    pub port: u16,
    /// Whether to accept LAN connections (binds `0.0.0.0`). Requires the
    /// `debug-server-lan` feature and defaults to false otherwise.
    pub allow_lan: bool,
}

impl Default for DebugServerOpts {
    fn default() -> Self {
        Self {
            port: 8080,
            allow_lan: cfg!(feature = "debug-server-lan"),
        }
    }
}

/// Handle to a running debug server instance.
pub struct DebugServerHandle {
    runtime: Option<Runtime>,
    server_task: Option<JoinHandle<Result<(), DebugServerError>>>,
    shutdown_tx: Option<oneshot::Sender<()>>,
    bound_addr: SocketAddr,
}

impl DebugServerHandle {
    /// Returns the address the server is bound to.
    pub fn bound_addr(&self) -> SocketAddr {
        self.bound_addr
    }
}

impl fmt::Debug for DebugServerHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DebugServerHandle")
            .field("bound_addr", &self.bound_addr)
            .finish()
    }
}

impl Drop for DebugServerHandle {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }

        if let (Some(runtime), Some(join_handle)) = (self.runtime.take(), self.server_task.take()) {
            let _ = runtime.block_on(async move {
                let _ = join_handle.await;
            });
        }
    }
}

/// Entry point for starting and stopping the embedded debug server.
pub struct DebugServer;

impl DebugServer {
    /// Spawn the debug server against the provided engine reference.
    pub fn start(
        engine: Arc<RwLock<ProcessingEngine>>,
        opts: DebugServerOpts,
    ) -> Result<DebugServerHandle, DebugServerError> {
        let runtime = RuntimeBuilder::new_multi_thread()
            .worker_threads(2)
            .thread_name("arbit-debug-server-worker")
            .enable_all()
            .build()
            .map_err(DebugServerError::RuntimeInit)?;

        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let (bound_tx, bound_rx) = oneshot::channel();

        let port = opts.port;
        let allow_lan = opts.allow_lan;
        let server_task = runtime.spawn(async move {
            let bind_ip = if allow_lan {
                IpAddr::V4(Ipv4Addr::UNSPECIFIED)
            } else {
                IpAddr::V4(Ipv4Addr::LOCALHOST)
            };
            let bind_addr = SocketAddr::new(bind_ip, port);

            let listener = tokio::net::TcpListener::bind(bind_addr)
                .await
                .map_err(|source| DebugServerError::Bind { addr: bind_addr, source })?;

            let local_addr = listener
                .local_addr()
                .map_err(DebugServerError::LocalAddr)?;

            let _ = bound_tx.send(local_addr);

            if allow_lan {
                tracing::warn!(%local_addr, "Debug server listening on LAN without authentication; intended for local dev only.");
            } else {
                tracing::info!(%local_addr, "Debug server listening on loopback interface");
            }

            let app: Router = router::build_router(engine);

            axum::serve(
                listener,
                app.into_make_service_with_connect_info::<SocketAddr>(),
            )
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await
            .map_err(DebugServerError::Serve)
        });

        let bound_addr = match runtime.block_on(bound_rx) {
            Ok(addr) => addr,
            Err(_) => {
                let err = runtime.block_on(async {
                    match server_task.await {
                        Ok(Ok(())) => DebugServerError::Startup(
                            "debug server exited before completing startup".into(),
                        ),
                        Ok(Err(err)) => err,
                        Err(join_err) => DebugServerError::Join(join_err),
                    }
                });
                return Err(err);
            }
        };

        tracing::debug!(%bound_addr, "Debug server started");

        Ok(DebugServerHandle {
            runtime: Some(runtime),
            server_task: Some(server_task),
            shutdown_tx: Some(shutdown_tx),
            bound_addr,
        })
    }

    /// Request a graceful shutdown of the running server.
    pub fn stop(mut handle: DebugServerHandle) -> Result<(), DebugServerError> {
        if let Some(tx) = handle.shutdown_tx.take() {
            let _ = tx.send(());
        }

        if let Some(runtime) = handle.runtime.take() {
            if let Some(join_handle) = handle.server_task.take() {
                match runtime.block_on(async move { join_handle.await }) {
                    Ok(Ok(())) => Ok(()),
                    Ok(Err(err)) => Err(err),
                    Err(join_err) => Err(DebugServerError::Join(join_err)),
                }
            } else {
                Ok(())
            }
        } else {
            Ok(())
        }
    }
}

/// Errors surfaced by the debug server runtime.
#[derive(Debug, thiserror::Error)]
pub enum DebugServerError {
    #[error("failed to initialize tokio runtime: {0}")]
    RuntimeInit(#[source] std::io::Error),
    #[error("failed to bind debug server at {addr}: {source}")]
    Bind {
        addr: SocketAddr,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to determine bound address: {0}")]
    LocalAddr(#[source] std::io::Error),
    #[error("debug server terminated during startup: {0}")]
    Startup(String),
    #[error("debug server error: {0}")]
    Serve(#[source] std::io::Error),
    #[error("debug server join error: {0}")]
    Join(#[from] tokio::task::JoinError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    #[test]
    fn default_opts_match_expectations() {
        let opts = DebugServerOpts::default();
        assert_eq!(opts.port, 8080);
        assert_eq!(opts.allow_lan, cfg!(feature = "debug-server-lan"));
    }

    #[test]
    fn start_and_stop_smoke() {
        let engine = Arc::new(RwLock::new(ProcessingEngine::new()));
        let opts = DebugServerOpts {
            port: 0,
            allow_lan: false,
        };

        let handle = DebugServer::start(engine, opts).expect("server should start");
        assert_eq!(handle.bound_addr().ip(), IpAddr::V4(Ipv4Addr::LOCALHOST));

        DebugServer::stop(handle).expect("server should stop cleanly");
    }
}
