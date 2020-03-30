extern crate num;

use num::Float;

pub mod record_io;
pub mod sparse;
pub mod word_vectors;

/// This function
pub fn estimate_max_prod<F: Float>(limit: F, tolerance: F) -> usize {
    let mut n: F = F::from(1e5).unwrap();
    let c: F = F::from(0.154_431_33).unwrap();
    while (limit - n * (n.ln() + c)).abs() > tolerance {
        n = limit / (n.ln() + c);
    }
    n.to_usize().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate() {
        let n = estimate_max_prod(1024.0, 1e-3);
        assert_eq!(n, 189);
    }
}
