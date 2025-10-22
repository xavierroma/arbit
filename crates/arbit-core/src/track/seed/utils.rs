use super::FeatureSeed;

pub fn radius_nms(mut seeds: Vec<FeatureSeed>, radius: f32, max_keep: usize) -> Vec<FeatureSeed> {
    let r2 = radius * radius;
    let mut kept: Vec<FeatureSeed> = Vec::with_capacity(seeds.len());
    'outer: for s in seeds.drain(..) {
        for k in &kept {
            let dx = s.position.x - k.position.x;
            let dy = s.position.y - k.position.y;
            if dx * dx + dy * dy <= r2 {
                continue 'outer;
            }
        }
        kept.push(s);
        if kept.len() == max_keep {
            break;
        }
    }
    kept
}
