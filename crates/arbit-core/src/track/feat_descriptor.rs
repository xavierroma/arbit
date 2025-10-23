use crate::img::pyramid::Pyramid;
use crate::track::FeatureSeed;

pub trait DescriptorBuffer: Clone {
    fn as_bytes(&self) -> &[u8];
}

impl DescriptorBuffer for [u8; 32] {
    fn as_bytes(&self) -> &[u8] {
        self
    }
}

impl DescriptorBuffer for Vec<u8> {
    fn as_bytes(&self) -> &[u8] {
        self
    }
}

#[derive(Debug, Clone)]
pub struct FeatDescriptor<D>
where
    D: DescriptorBuffer,
{
    pub seed: FeatureSeed,
    pub angle: f32,
    pub data: D,
}

impl<D> FeatDescriptor<D>
where
    D: DescriptorBuffer,
{
    pub fn bytes(&self) -> &[u8] {
        self.data.as_bytes()
    }
}

pub trait FeatDescriptorExtractor {
    type Storage: DescriptorBuffer;

    const LEN: usize;

    fn describe(
        &self,
        pyramid: &Pyramid,
        seeds: &[FeatureSeed],
    ) -> Vec<FeatDescriptor<Self::Storage>>;
}

pub mod orb;
pub use orb::OrbDescriptor;
