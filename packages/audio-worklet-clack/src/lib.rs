//! Ball↔ball collision "clack", modeled on a MEASURED billiard impact (a CC0
//! recording, freesound #539854, analyzed offline — see BOX3D.md notes):
//!
//! * Energy lives almost entirely in **2–4 kHz** (88%) + 1–2 kHz (11%);
//!   below 1 kHz there is essentially nothing (0.7%) — no "body thump".
//! * The spectrum is a **dense cluster** of many comparable peaks packed into
//!   ~1.8–2.9 kHz — too dense to hear as a pitch (that's what separates
//!   "ceramic klak" from "struck metal pan").
//! * The envelope drops −20 dB in ~6 ms, then a quiet 2 kHz tail rings to
//!   ~90 ms. The 4–8 kHz content exists only in the first millisecond.
//! * The balls **rattle**: measured re-contacts at ~12/17/23/30 ms with
//!   decaying amplitude — a big part of why two real balls sound "real".
//!
//! Synthesis mirrors those measurements: a Hertzian contact pulse (duration
//! shrinks with impact speed) + a burst of low-passed contact noise excite a
//! dense 8-mode cluster (fast decay) + 2 mid modes + 2 quiet slow tail modes;
//! micro-bounce re-excitations are scheduled per the measured rattle, gated
//! by intensity (a gentle touch is a single contact). Each voice seeds its
//! own jitter (mode spacing, bounce timing) so no two clacks are identical.
//!
//! Params: `intensity` is the RUNTIME control (the game overrides it per
//! collision); `brightness` slides the cluster center, `ring` scales decays.

use core::sync::atomic::{AtomicU32, Ordering};

use awsm_audio_worklet::{awsm_worklet, math, ParamDesc, Params, Processor};

const TWO_PI: f32 = core::f32::consts::TAU;

/// Per-voice seed so consecutive clacks differ (single audio thread; relaxed
/// is fine — this is entropy, not synchronization).
static VOICE_COUNTER: AtomicU32 = AtomicU32::new(0x9e37_79b9);

/// Measured micro-bounce schedule: (time ms, relative amplitude). Jittered
/// per voice; the number that actually fire scales with intensity.
const BOUNCES: [(f32, f32); 4] = [(11.5, 0.10), (16.8, 0.07), (23.0, 0.05), (30.0, 0.035)];

/// One 2-pole resonator: `y[n] = 2r·cos(ω)·y[n-1] − r²·y[n-2] + x[n]`.
#[derive(Clone, Copy, Default)]
struct Mode {
    b1: f32,
    b2: f32,
    y1: f32,
    y2: f32,
    gain: f32,
}

impl Mode {
    fn set(&mut self, freq_hz: f32, tau_secs: f32, gain: f32, sr: f32) {
        // Latch-time only — std float math (precise coefficients matter).
        let f = freq_hz.min(sr * 0.42);
        let r = (-1.0 / (tau_secs * sr)).exp();
        let w = TWO_PI * f / sr;
        self.b1 = 2.0 * r * w.cos();
        self.b2 = -r * r;
        self.gain = gain;
    }

    #[inline]
    fn tick(&mut self, x: f32) -> f32 {
        let y = self.b1 * self.y1 + self.b2 * self.y2 + x;
        self.y2 = self.y1;
        self.y1 = y;
        y * self.gain
    }
}

/// Tiny deterministic LCG (no allocation, no OS).
struct Lcg(u32);

impl Lcg {
    #[inline]
    fn next_bi(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        ((self.0 >> 8) as f32 / 8_388_608.0) - 1.0
    }
    /// Uniform in [1-j, 1+j].
    #[inline]
    fn jitter(&mut self, j: f32) -> f32 {
        1.0 + self.next_bi() * j
    }
}

/// One scheduled contact: a raised-cosine pulse + noise burst from `start`
/// (in samples).
#[derive(Clone, Copy, Default)]
struct Contact {
    start: f32,
    len: f32,
    amp: f32,
}

struct Clack {
    sr: f32,
    n: u32,
    latched: bool,
    /// Initial hit + up to 4 micro-bounces (measured rattle).
    contacts: [Contact; 5],
    /// 12 modes: 8 dense cluster + 2 mid + 2 quiet slow tail.
    modes: [Mode; 12],
    rng: Lcg,
    grit_amp: f32,
    /// One-pole low-pass for the contact noise (the real 4–8 kHz content
    /// exists only in the first ms — unfiltered white noise reads tinny).
    noise_lp: f32,
    noise_a: f32,
    out_scale: f32,
}

impl Processor for Clack {
    const PARAMS: &'static [ParamDesc] = &[
        ParamDesc::new("intensity", 0.0, 1.0, 0.7),
        ParamDesc::new("brightness", 0.0, 1.0, 0.5),
        ParamDesc::new("ring", 0.25, 2.0, 1.0),
    ];

    fn new(sample_rate: f32) -> Self {
        let seed = VOICE_COUNTER.fetch_add(0x6d2b_79f5, Ordering::Relaxed);
        Clack {
            sr: sample_rate,
            n: 0,
            latched: false,
            contacts: [Contact::default(); 5],
            modes: [Mode::default(); 12],
            rng: Lcg(seed),
            grit_amp: 0.0,
            noise_lp: 0.0,
            noise_a: 0.3,
            out_scale: 1.0,
        }
    }

    fn process(&mut self, _input: &[&[f32]], output: &mut [&mut [f32]], params: &Params) {
        if !self.latched {
            let intensity = params.get(0).clamp(0.0, 1.0);
            let brightness = params.get(1).clamp(0.0, 1.0);
            let ring = params.get(2).clamp(0.25, 2.0);

            // ── Contacts: the hit + intensity-gated micro-bounces ──────────
            // Hertzian pulse, SHORTER when harder (0.9 → 0.3 ms).
            let contact_secs = 0.0005 - 0.0003 * intensity;
            let base_len = (contact_secs * self.sr).max(2.0);
            let amp = 0.25 + 0.75 * intensity.powf(1.3);
            self.contacts[0] = Contact {
                start: 0.0,
                len: base_len,
                amp,
            };
            // A gentle touch is ONE contact; a hard hit rattles (measured
            // bumps at ~12/17/23/30 ms, decaying) — jittered so every
            // collision differs.
            let n_bounces = (intensity * 4.5) as usize; // 0..=4
            for (i, (ms, rel)) in BOUNCES.iter().enumerate() {
                if i < n_bounces {
                    self.contacts[i + 1] = Contact {
                        start: ms / 1000.0 * self.sr * self.rng.jitter(0.18),
                        // Re-contacts are softer → longer pulses.
                        len: base_len * 1.2,
                        amp: amp * rel * self.rng.jitter(0.3),
                    };
                } else {
                    self.contacts[i + 1] = Contact::default();
                }
            }

            // Contact noise: only the first ~1.5 ms per contact (the measured
            // 4–8 kHz content), low-passed so it's a "k", not fizz.
            self.grit_amp = 0.3 + 0.5 * intensity;
            let noise_cut = 3200.0 + 1200.0 * brightness;
            self.noise_a = 1.0 - (-TWO_PI * noise_cut / self.sr).exp();

            // ── The measured spectrum ───────────────────────────────────────
            // Dense cluster: 8 modes packed into ~1.8–2.9 kHz (center slides
            // ±400 Hz with brightness), fast decay (−20 dB ≈ 6 ms), spacing
            // jittered per voice so the cluster never reads as a chord.
            let center = 2400.0 + 700.0 * (brightness - 0.5);
            let spread = 280.0;
            for k in 0..8 {
                let frac = (k as f32 / 7.0) * 2.0 - 1.0; // -1..1
                let f = center + frac * spread * self.rng.jitter(0.12);
                let tau = 0.0034 * self.rng.jitter(0.3);
                self.modes[k].set(f, tau * ring, 1.0, self.sr);
            }
            // Tail cluster: the measured late window is a dense quiet ring
            // across ~2.1–2.6 kHz to ~90 ms (4 modes so it never reads as a
            // bell chord).
            for (slot, (f, g)) in [
                (2100.0, 0.34),
                (2250.0, 0.32),
                (2400.0, 0.28),
                (2600.0, 0.22),
            ]
            .iter()
            .enumerate()
            {
                self.modes[8 + slot].set(
                    f * self.rng.jitter(0.03),
                    0.028 * ring * self.rng.jitter(0.2),
                    *g,
                    self.sr,
                );
            }

            // Calibrated so intensity=1 peaks ≈ 0.7 pre-panner without
            // saturating the tanh ceiling (HRTF adds up to +6 dB).
            self.out_scale = 0.022 * amp;
            self.latched = true;
        }

        let frames = output.first().map(|c| c.len()).unwrap_or(0);
        for i in 0..frames {
            let t = self.n as f32;
            let mut x = 0.0;
            for c in self.contacts.iter() {
                if c.amp > 0.0 && t >= c.start && t < c.start + c.len {
                    let phase = (t - c.start) / c.len;
                    let pulse =
                        0.5 - 0.5 * math::sin(TWO_PI * phase + core::f32::consts::FRAC_PI_2);
                    x += pulse * c.amp;
                }
                // Each contact carries its own short noise burst.
                let nl = 0.0015 * self.sr;
                let nt = t - c.start;
                if c.amp > 0.0 && nt >= 0.0 && nt < nl {
                    let fall = 1.0 - nt / nl;
                    self.noise_lp += self.noise_a * (self.rng.next_bi() - self.noise_lp);
                    x += self.noise_lp * self.grit_amp * c.amp * fall * fall;
                }
            }

            let mut s = 0.0;
            for m in self.modes.iter_mut() {
                s += m.tick(x);
            }
            // Safety ceiling, NOT an effect — must stay un-saturated.
            let y = math::tanh(s * self.out_scale);

            for channel in output.iter_mut() {
                channel[i] = y;
            }
            self.n = self.n.wrapping_add(1);
        }
    }
}

awsm_worklet!(Clack);
