//! The render worker's side of the Stats panel's breakdowns, measured only
//! while main asks (the `DETAIL_*` bits in [`BodyMotion::detail`]):
//!
//! * `DETAIL_RENDERER`: on every frame, the phases of our own frame function
//!   are timed (one clock read between each) and awsm-renderer's CPU scopes
//!   are on (`logging.cpu = SubFrame`, a field the renderer checks at every
//!   scope). Otherwise the tier is `Off` and nothing reads the clock.
//! * `DETAIL_GPU`: the per-pass GPU times are passed on. Measuring them is
//!   already paid for: the frame total the panel always shows comes from the
//!   same timestamp queries (`logging.gpu = Frame` times every pass).
//!
//! The numbers go to main as running totals in a [`RenderMsg::Timings`], a
//! few times a second. Every renderer entry counts the frames timed, so main's
//! difference of two totals is a mean per frame drawn while the group was open.
//!
//! [`BodyMotion::detail`]: crate::protocol::BodyMotion::detail
//! [`RenderMsg::Timings`]: crate::protocol::RenderMsg::Timings

use awsm_renderer::debug::TimingTier;
use awsm_renderer::profiling::{self, CpuScopeKind};
use awsm_renderer::AwsmRenderer;

use crate::protocol::{TimingKind, TimingTotal, DETAIL_GPU, DETAIL_RENDERER};

/// The phases of one frame, in the order they run; the Stats panel lists
/// them by these names.
#[derive(Clone, Copy)]
pub enum Phase {
    /// Everything up to the transforms being written: waking physics, the
    /// camera ring, sampling + interpolating the published steps, minting
    /// dropped balls.
    ReadApp,
    SetCamera,
    UpdateTransforms,
    /// `AwsmRenderer::render`: encoding every pass and submitting.
    Render,
}

pub const PHASE_NAMES: [&str; 4] = [
    "read app state + interpolate",
    "set camera",
    "update transforms",
    "render (encode + submit)",
];

/// How often the totals go to main, in milliseconds.
const PUBLISH_EVERY_MS: f64 = 250.0;

#[derive(Default)]
pub struct Profile {
    /// What main asked for at the start of the current frame.
    detail: u32,
    published_at_ms: f64,
    /// Running totals (ms) of each phase over the frames timed, and those
    /// frames.
    phase_ms: [f64; 4],
    frames: u64,
}

impl Profile {
    fn renderer(&self) -> bool {
        self.detail & DETAIL_RENDERER != 0
    }

    /// Starts a frame: picks up what main asks for and switches the
    /// renderer's CPU scopes on or off for it.
    pub fn begin_frame(&mut self, renderer: &mut AwsmRenderer, detail: u32) {
        self.detail = detail;
        renderer.logging.cpu = if self.renderer() {
            TimingTier::SubFrame
        } else {
            TimingTier::Off
        };
    }

    /// While timing: charges the time since `since_ms` to `phase` and returns
    /// the time now, for the next phase to start from.
    pub fn lap(&mut self, phase: Phase, perf: &web_sys::Performance, since_ms: f64) -> f64 {
        if !self.renderer() {
            return since_ms;
        }
        let now = perf.now();
        self.phase_ms[phase as usize] += now - since_ms;
        now
    }

    /// Ends a frame: counts it, and every [`PUBLISH_EVERY_MS`] returns the
    /// totals for main (`None` otherwise, and always while nothing is asked).
    pub fn end_frame(&mut self, frame_ms: f64) -> Option<Vec<TimingTotal>> {
        let renderer = self.renderer();
        let gpu = self.detail & DETAIL_GPU != 0;
        if renderer {
            self.frames += 1;
        }
        if !(renderer || gpu) || (frame_ms - self.published_at_ms).abs() < PUBLISH_EVERY_MS {
            return None;
        }
        self.published_at_ms = frame_ms;

        let mut entries = Vec::new();
        if renderer {
            let frames = self.frames as f64;
            for (name, ms) in PHASE_NAMES.into_iter().zip(self.phase_ms) {
                entries.push(total(TimingKind::RendererPhase, name, ms, frames));
            }
            // The renderer's totals only grow while the tier is on, i.e. on
            // the frames counted here.
            for (name, stat) in profiling::cpu_timing_stats() {
                let (kind, name) = match profiling::cpu_scope_kind(name) {
                    // The whole of render(): our render phase already.
                    CpuScopeKind::Frame => continue,
                    CpuScopeKind::Prep => (
                        TimingKind::RendererPrep,
                        name.trim_start_matches(profiling::CPU_PREP_PREFIX),
                    ),
                    CpuScopeKind::Pass => (TimingKind::RendererPass, name),
                    CpuScopeKind::Submit => (
                        TimingKind::RendererSubmit,
                        name.trim_start_matches(profiling::CPU_SUBMIT_PREFIX),
                    ),
                    CpuScopeKind::Hook => (TimingKind::RendererHook, name),
                };
                entries.push(total(kind, name, stat.total_ms, frames));
            }
        }
        if gpu {
            // Timestamps are read back for some frames only; the whole-frame
            // entry counts exactly those.
            let sampled = profiling::gpu_frame_timing().map_or(0, |frame| frame.count) as f64;
            for (name, stat) in profiling::gpu_timing_stats() {
                if name != profiling::GPU_FRAME_SCOPE {
                    entries.push(total(TimingKind::GpuPass, name, stat.total_ms, sampled));
                }
            }
        }
        Some(entries)
    }
}

fn total(kind: TimingKind, name: &str, total_ms: f64, count: f64) -> TimingTotal {
    TimingTotal {
        kind,
        name: name.to_string(),
        total_ms,
        count,
    }
}
