use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum NativeKernelError {
    #[error("feature '{0}' is disabled")]
    FeatureDisabled(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelStatus {
    Ready,
    Disabled,
}

#[derive(Debug, Default, Clone)]
pub struct OpenCvKernelAdapter;

impl OpenCvKernelAdapter {
    pub fn status(&self) -> KernelStatus {
        if cfg!(feature = "native-opencv") {
            KernelStatus::Ready
        } else {
            KernelStatus::Disabled
        }
    }

    pub fn warmup(&self) -> Result<(), NativeKernelError> {
        if cfg!(feature = "native-opencv") {
            Ok(())
        } else {
            Err(NativeKernelError::FeatureDisabled("native-opencv"))
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct GtsamKernelAdapter;

impl GtsamKernelAdapter {
    pub fn status(&self) -> KernelStatus {
        if cfg!(feature = "native-gtsam") {
            KernelStatus::Ready
        } else {
            KernelStatus::Disabled
        }
    }

    pub fn warmup(&self) -> Result<(), NativeKernelError> {
        if cfg!(feature = "native-gtsam") {
            Ok(())
        } else {
            Err(NativeKernelError::FeatureDisabled("native-gtsam"))
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct BowKernelAdapter;

impl BowKernelAdapter {
    pub fn status(&self) -> KernelStatus {
        if cfg!(feature = "native-bow") {
            KernelStatus::Ready
        } else {
            KernelStatus::Disabled
        }
    }

    pub fn warmup(&self) -> Result<(), NativeKernelError> {
        if cfg!(feature = "native-bow") {
            Ok(())
        } else {
            Err(NativeKernelError::FeatureDisabled("native-bow"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapters_report_disabled_without_features() {
        let opencv = OpenCvKernelAdapter;
        let gtsam = GtsamKernelAdapter;
        let bow = BowKernelAdapter;

        assert_eq!(opencv.status(), KernelStatus::Disabled);
        assert_eq!(gtsam.status(), KernelStatus::Disabled);
        assert_eq!(bow.status(), KernelStatus::Disabled);

        assert!(opencv.warmup().is_err());
        assert!(gtsam.warmup().is_err());
        assert!(bow.warmup().is_err());
    }
}
