//! Debug server surface for the processing engine.
//!
//! The full implementation lives behind the `debug-server` feature flag. At
//! this stage, the module only defines the public API surface used by the
//! runtime and higher layers.

use std::sync::{Arc, RwLock};

use crate::ProcessingEngine;

/// Configuration options controlling the debug server runtime.
#[derive(Debug, Clone)]
pub struct DebugServerOpts {
    pub port: u16,
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

/// Running debug server instance handle.
#[derive(Debug)]
pub struct DebugServerHandle;

/// Entry point for starting the embedded debug server.
#[derive(Debug)]
pub struct DebugServer;

impl DebugServer {
    /// Spawn the debug server against the provided engine reference.
    pub fn start(
        _engine: Arc<RwLock<ProcessingEngine>>,
        _opts: DebugServerOpts,
    ) -> Result<DebugServerHandle, DebugServerError> {
        Err(DebugServerError::NotImplemented)
    }

    /// Request a graceful shutdown of the running server.
    pub fn stop(_handle: DebugServerHandle) {}
}

/// Errors surfaced by the debug server.
#[derive(Debug, thiserror::Error)]
pub enum DebugServerError {
    #[error("debug server is not implemented yet")]
    NotImplemented,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_opts_match_expectations() {
        let opts = DebugServerOpts::default();
        assert_eq!(opts.port, 8080);
        assert_eq!(opts.allow_lan, cfg!(feature = "debug-server-lan"));
    }
}
