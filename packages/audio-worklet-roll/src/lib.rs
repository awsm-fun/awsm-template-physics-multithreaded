//! Rolling-ball DSP worklet for awsm-audio.
//!
//! Model: a rolling object is a stream of micro-impacts against surface
//! irregularities, each exciting the resonant body of the ball + table. The
//! *rate* of those impacts scales with rolling speed — sparse discrete ticks at
//! a crawl, a dense continuous rumble at speed. Built-in noise nodes can't do
//! this (their grain density is fixed at authoring time); here `speed` drives a
//! Poisson impulse train at audio rate, feeding three modal resonators.
//!
//! Params (all k-rate, written live by the game):
//!   0 speed      0..1   impact rate + excitation energy + ring length
//!   1 roughness  0..1   surface grain: impact amplitude spread
//!   2 body_hz   40..320 fundamental resonance of the rolling body
//!   3 brightness 0..1   how much the higher modes open up
//!   4 gain       0..2   output trim

use awsm_audio_worklet::{awsm_worklet, math, ParamDesc, Params, Processor};

const TWO_PI: f32 = core::f32::consts::PI * 2.0;
const HALF_PI: f32 = core::f32::consts::FRAC_PI_2;

/// Two-pole ringing resonator: y[n] = x[n] + a1·y[n-1] − a2·y[n-2],
/// with a1 = 2r·cos(θ), a2 = r². Rings at θ; r (<1) sets decay/Q.
struct Reso {
    y1: f32,
    y2: f32,
    a1: f32,
    a2: f32,
    norm: f32,
}

impl Reso {
    fn new() -> Self {
        Reso {
            y1: 0.0,
            y2: 0.0,
            a1: 0.0,
            a2: 0.0,
            norm: 0.0,
        }
    }

    #[inline]
    fn set(&mut self, freq: f32, r: f32, sr: f32) {
        let theta = TWO_PI * freq / sr;
        // cos(theta) via the crate's sin (avoids pulling f32::cos into the wasm).
        let cos_theta = math::sin(theta + HALF_PI);
        self.a1 = 2.0 * r * cos_theta;
        self.a2 = r * r;
        // Bandwidth normalization: this 2-pole has peak gain ~1/(1−r), so scaling
        // the input by (1−r) keeps steady-state output near unity regardless of Q.
        // Without it the resonators build to tens and slam tanh into a buzzy square.
        self.norm = 1.0 - r;
    }

    #[inline]
    fn tick(&mut self, x: f32) -> f32 {
        let y = self.norm * x + self.a1 * self.y1 - self.a2 * self.y2;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

struct RollWorklet {
    sr: f32,
    rng: u32,
    r1: Reso,
    r2: Reso,
    r3: Reso,
    exc_lp: f32, // one-pole smoothing of the excitation (rounds off clicky grit)
    out_lp: f32, // one-pole smoothing of the output (tames residual high-freq grit)
    dc_x1: f32,
    dc_y1: f32,
}

impl RollWorklet {
    #[inline]
    fn rand_u(&mut self) -> u32 {
        // xorshift32 — cheap, allocation-free, deterministic.
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.rng = x;
        x
    }

    #[inline]
    fn rand_f(&mut self) -> f32 {
        (self.rand_u() >> 8) as f32 / 16_777_216.0
    }

    #[inline]
    fn rand_bi(&mut self) -> f32 {
        self.rand_f() * 2.0 - 1.0
    }
}

impl Processor for RollWorklet {
    const PARAMS: &'static [ParamDesc] = &[
        ParamDesc::new("speed", 0.0, 1.0, 0.3),
        ParamDesc::new("roughness", 0.0, 1.0, 0.5),
        ParamDesc::new("body_hz", 40.0, 320.0, 110.0),
        ParamDesc::new("brightness", 0.0, 1.0, 0.5),
        ParamDesc::new("gain", 0.0, 2.0, 1.0),
    ];

    fn new(sample_rate: f32) -> Self {
        RollWorklet {
            sr: sample_rate,
            rng: 0x1234_5678,
            r1: Reso::new(),
            r2: Reso::new(),
            r3: Reso::new(),
            exc_lp: 0.0,
            out_lp: 0.0,
            dc_x1: 0.0,
            dc_y1: 0.0,
        }
    }

    fn process(&mut self, _input: &[&[f32]], output: &mut [&mut [f32]], params: &Params) {
        let speed = clamp01(params.get(0));
        let roughness = clamp01(params.get(1));
        let body = params.get(2).max(20.0);
        let brightness = clamp01(params.get(3));
        let gain = params.get(4);

        // Long rings = a smooth, sustained rumble rather than ticky grit.
        // Kept < 1 for stability.
        self.r1.set(body, 0.992 + 0.006 * speed, self.sr);
        self.r2.set(body * 2.13, 0.990 + 0.007 * speed, self.sr);
        self.r3.set(body * 3.46, 0.986 + 0.008 * speed, self.sr);

        // Impact rate: sparse at low speed, dense at high speed.
        let p_impact = (12.0 + speed * speed * 650.0) / self.sr;
        let bed_amp = 0.005 + speed * 0.025; // quiet floor between impacts
        let hit_amp = 0.20 + speed * 0.70; // per-impact excitation energy

        // One-pole smoothing coefficients (a = 2π·fc/sr, clamped). Low-passing the
        // excitation turns the sharp impulses into soft rolling contacts (kills the
        // clicky grit); a gentle output low-pass shaves any residual fizz.
        // Brightness opens both filters up.
        let exc_a = (TWO_PI * (160.0 + brightness * 600.0) / self.sr).min(1.0);
        let out_a = (TWO_PI * (1100.0 + brightness * 2200.0) / self.sr).min(1.0);

        // Upper modes kept low for a rounder, less buzzy timbre.
        let w2 = 0.20 + 0.28 * brightness;
        let w3 = 0.05 + 0.16 * brightness;
        // Higher base scale compensates for the LP'd excitation + longer rings;
        // trim with the `gain` param (no recompile).
        let out_scale = 8.0 * gain;

        let frames = output.first().map(|c| c.len()).unwrap_or(0);
        for n in 0..frames {
            // Continuous low excitation keeps the body alive between impacts.
            let mut e = self.rand_bi() * bed_amp;
            // Poisson impact: occasional grain, amplitude spread by roughness.
            if self.rand_f() < p_impact {
                let spread = (1.0 - roughness) + roughness * self.rand_f() * 2.0;
                e += self.rand_bi() * hit_amp * spread;
            }
            // Smooth the excitation (clicks → soft contacts).
            self.exc_lp += exc_a * (e - self.exc_lp);
            let e = self.exc_lp;
            // Excite the modal bank.
            let s = self.r1.tick(e) + self.r2.tick(e) * w2 + self.r3.tick(e) * w3;
            // DC blocker (resonators can accumulate a small offset).
            let dc = s - self.dc_x1 + 0.997 * self.dc_y1;
            self.dc_x1 = s;
            self.dc_y1 = dc;
            // Gentle output low-pass, then soft-clip ceiling.
            self.out_lp += out_a * (dc - self.out_lp);
            let y = math::tanh(self.out_lp * out_scale);
            for channel in output.iter_mut() {
                channel[n] = y;
            }
        }
    }
}

#[inline]
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

awsm_worklet!(RollWorklet);
