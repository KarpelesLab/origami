//! Tiny deterministic PRNG for the Langevin integrator.
//!
//! `Xoshiro256pp` is Blackman & Vigna's xoshiro256++ generator: 256-bit
//! state, ~10ns per draw, passes BigCrush. We seed the four 64-bit state
//! words from a single 64-bit user seed via SplitMix64 (the recommended
//! bootstrap) so the user-facing API is one `u64`.
//!
//! Gaussian samples come from a two-output Box-Muller transform. The
//! second value is cached so the per-draw cost is one square-root + one
//! sin/cos amortised over two draws — fine for the integrator's appetite
//! of 3N gaussians per step.
//!
//! No `rand` / `rand_distr` dependency: this is ~80 lines and reproduces
//! identically across machines.

/// xoshiro256++ generator with cached spare Gaussian.
#[derive(Debug, Clone)]
pub struct Xoshiro256pp {
    state: [u64; 4],
    cached_gaussian: Option<f64>,
}

impl Xoshiro256pp {
    /// Construct from a single 64-bit seed. The four state words are
    /// derived via SplitMix64 starting from `seed`. All-zero state is
    /// avoided because xoshiro requires a non-zero starting state.
    pub fn from_seed(seed: u64) -> Self {
        let mut sm = SplitMix64 { state: seed };
        let s0 = sm.next();
        let s1 = sm.next();
        let s2 = sm.next();
        let s3 = sm.next();
        Self {
            state: [s0, s1, s2, s3],
            cached_gaussian: None,
        }
    }

    /// Raw 64-bit output of xoshiro256++.
    pub fn next_u64(&mut self) -> u64 {
        let result = self.state[0]
            .wrapping_add(self.state[3])
            .rotate_left(23)
            .wrapping_add(self.state[0]);

        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);

        result
    }

    /// Uniform double in `[0, 1)` with 53 bits of resolution.
    pub fn next_f64(&mut self) -> f64 {
        // The standard recipe: take the high 53 bits and scale by 2^-53.
        ((self.next_u64() >> 11) as f64) * (1.0 / (1u64 << 53) as f64)
    }

    /// Standard normal sample, mean 0, variance 1. Uses Box-Muller with
    /// a cached second value to halve the cost of trig calls.
    pub fn gaussian(&mut self) -> f64 {
        if let Some(g) = self.cached_gaussian.take() {
            return g;
        }
        // Avoid log(0) by rejecting u1 = 0 (probability 2^-53; loop is
        // a single iteration in practice).
        let mut u1 = self.next_f64();
        while u1 <= 0.0 {
            u1 = self.next_f64();
        }
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        let (sin_t, cos_t) = theta.sin_cos();
        self.cached_gaussian = Some(r * sin_t);
        r * cos_t
    }
}

/// Internal helper used only to derive an xoshiro initial state from a
/// single user seed. Not exported.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_mean_and_variance() {
        let mut rng = Xoshiro256pp::from_seed(42);
        const N: usize = 100_000;
        let mut sum = 0.0;
        let mut sum2 = 0.0;
        let mut max = f64::NEG_INFINITY;
        let mut min = f64::INFINITY;
        for _ in 0..N {
            let u = rng.next_f64();
            assert!((0.0..1.0).contains(&u), "uniform out of range: {u}");
            sum += u;
            sum2 += u * u;
            if u > max {
                max = u;
            }
            if u < min {
                min = u;
            }
        }
        let mean = sum / N as f64;
        let variance = sum2 / N as f64 - mean * mean;
        // Expected mean 0.5, variance 1/12 ≈ 0.0833.
        assert!((mean - 0.5).abs() < 0.01, "mean {mean} far from 0.5");
        assert!((variance - 1.0 / 12.0).abs() < 0.002, "variance {variance} far from 1/12");
    }

    #[test]
    fn gaussian_mean_and_variance() {
        let mut rng = Xoshiro256pp::from_seed(7);
        const N: usize = 200_000;
        let mut sum = 0.0;
        let mut sum2 = 0.0;
        for _ in 0..N {
            let g = rng.gaussian();
            sum += g;
            sum2 += g * g;
        }
        let mean = sum / N as f64;
        let variance = sum2 / N as f64 - mean * mean;
        // Standard normal: mean 0, variance 1.
        assert!(mean.abs() < 0.02, "Gaussian mean {mean} far from 0");
        assert!((variance - 1.0).abs() < 0.02, "Gaussian variance {variance} far from 1");
    }

    #[test]
    fn reproducible_with_same_seed() {
        let mut a = Xoshiro256pp::from_seed(123);
        let mut b = Xoshiro256pp::from_seed(123);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn distinct_seeds_diverge() {
        let mut a = Xoshiro256pp::from_seed(1);
        let mut b = Xoshiro256pp::from_seed(2);
        let mut differs = false;
        for _ in 0..16 {
            if a.next_u64() != b.next_u64() {
                differs = true;
                break;
            }
        }
        assert!(differs, "two seeds produced identical streams");
    }
}
