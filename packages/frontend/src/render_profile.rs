//! The render worker's side of the Stats panel's breakdowns, measured only
//! while main asks (the `DETAIL_*` bits in [`BodyMotion::detail`]):
//!
//! * `DETAIL_RENDERER`: the phases of our own frame function are timed (one
//!   clock read between each), and awsm-renderer's per-pass CPU scopes are
//!   switched on for some frames (`logging.cpu`, a field the renderer checks
//!   at every scope).
//! * `DETAIL_GPU`: the per-pass GPU times are passed on. Measuring them is
//!   already paid for: the frame total the panel always shows comes from the
//!   same timestamp queries (`logging.gpu = Frame` times every pass).
//!
//! The numbers go to main as running totals in a [`RenderMsg::Timings`], a
//! few times a second.
//!
//! # The per-pass scopes are costly, so they are sampled
//!
//! Each awsm-renderer scope looks the clock up through the global object at
//! both ends, which in a worker takes far longer than most passes (about
//! 80 µs a lookup: a dozen scopes cost several times the frame). So that
//! looking does not change what is looked at:
//!
//! * the scopes are on for one frame in [`PASS_TIMING_EVERY`] only, and the
//!   phases are taken from the other frames, which are undisturbed;
//! * what the scopes cost is measured, as the difference between the render
//!   phase with and without them, and published as a row of its own. One
//!   estimate is made per publish and the median of the last few is used, so
//!   that a frame that hitched (a pipeline compile, a collection) does not
//!   pass for the scopes' doing;
//! * each pass has the scope's share of its reading taken out. A scope's
//!   reading runs from the end of its first lookup to the end of its second,
//!   so it contains one of the two: half of what the scope costs.
//!
//! [`BodyMotion::detail`]: crate::protocol::BodyMotion::detail
//! [`RenderMsg::Timings`]: crate::protocol::RenderMsg::Timings

use std::collections::{HashMap, VecDeque};

use awsm_renderer::debug::TimingTier;
use awsm_renderer::{profiling, AwsmRenderer};

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

/// awsm-renderer's own scope around the whole of `render()`: the same time
/// as our render phase, so it is not listed among the passes.
const WHOLE_RENDER_SCOPE: &str = "Render";
/// The name the cost of the scopes is published under.
pub const OVERHEAD_NAME: &str = "timing the passes";

/// The per-pass scopes are on for one frame in this many.
const PASS_TIMING_EVERY: u64 = 16;
/// How often the totals go to main, in milliseconds.
const PUBLISH_EVERY_MS: f64 = 250.0;
/// The scope cost in use is the median of this many estimates, one per
/// publish that had frames of both kinds: a few seconds' worth.
const SCOPE_ESTIMATES: usize = 15;

#[derive(Default)]
pub struct Profile {
    /// What main asked for at the start of the current frame, and whether
    /// the per-pass scopes are on for it.
    detail: u32,
    timing_passes: bool,
    published_at_ms: f64,

    /// Running totals (ms) over the frames WITHOUT the scopes: each phase,
    /// and the frames counted.
    phase_ms: [f64; 4],
    plain_frames: u64,
    /// The frames WITH them so far.
    timed_frames: u64,
    /// The render phase since the last publish, on frames of each kind.
    window: Window,

    /// Per scope: the renderer's totals at the last publish, and our running
    /// total with the scope's own share taken out.
    passes: HashMap<&'static str, PassTotal>,
    /// The latest estimates of what one scope costs (ms), and the running
    /// total of what the scopes cost by them.
    scope_estimates: VecDeque<f64>,
    overhead_ms: f64,
}

#[derive(Default)]
struct Window {
    plain_render_ms: f64,
    plain_frames: u32,
    timed_render_ms: f64,
    timed_frames: u32,
}

#[derive(Default)]
struct PassTotal {
    seen_ms: f64,
    seen_count: u64,
    net_ms: f64,
}

impl Profile {
    fn renderer(&self) -> bool {
        self.detail & DETAIL_RENDERER != 0
    }

    /// Starts a frame: picks up what main asks for and switches the
    /// renderer's per-pass CPU scopes on or off for it.
    pub fn begin_frame(&mut self, renderer: &mut AwsmRenderer, detail: u32) {
        self.detail = detail;
        let frames = self.plain_frames + self.timed_frames;
        self.timing_passes = self.renderer() && frames.is_multiple_of(PASS_TIMING_EVERY);
        renderer.logging.cpu = if self.timing_passes {
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
        if !self.timing_passes {
            self.phase_ms[phase as usize] += now - since_ms;
        }
        if matches!(phase, Phase::Render) {
            if self.timing_passes {
                self.window.timed_render_ms += now - since_ms;
            } else {
                self.window.plain_render_ms += now - since_ms;
            }
        }
        now
    }

    /// Ends a frame: counts it, and every [`PUBLISH_EVERY_MS`] returns the
    /// totals for main (`None` otherwise, and always while nothing is asked).
    pub fn end_frame(&mut self, frame_ms: f64) -> Option<Vec<TimingTotal>> {
        let renderer = self.renderer();
        let gpu = self.detail & DETAIL_GPU != 0;
        if self.timing_passes {
            self.timed_frames += 1;
            self.window.timed_frames += 1;
        } else if renderer {
            self.plain_frames += 1;
            self.window.plain_frames += 1;
        }
        if !(renderer || gpu) || (frame_ms - self.published_at_ms).abs() < PUBLISH_EVERY_MS {
            return None;
        }
        self.published_at_ms = frame_ms;

        let mut entries = Vec::new();
        if renderer {
            self.publish_renderer(&mut entries);
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

    fn publish_renderer(&mut self, entries: &mut Vec<TimingTotal>) {
        let plain_frames = self.plain_frames as f64;
        for (name, ms) in PHASE_NAMES.into_iter().zip(self.phase_ms) {
            entries.push(total(TimingKind::RendererPhase, name, ms, plain_frames));
        }

        // What the scopes ran and read since the last publish.
        let stats = profiling::cpu_timing_stats();
        let mut scopes = Vec::with_capacity(stats.len());
        let mut scope_runs = 0.0;
        for (name, stat) in stats {
            let pass = self.passes.entry(name).or_default();
            let runs = stat.count.saturating_sub(pass.seen_count) as f64;
            let read_ms = stat.total_ms - pass.seen_ms;
            pass.seen_count = stat.count;
            pass.seen_ms = stat.total_ms;
            scope_runs += runs;
            scopes.push((name, runs, read_ms));
        }

        // What one scope costs: what a frame with the scopes took over one
        // without, shared among the scopes it ran.
        let window = std::mem::take(&mut self.window);
        if window.plain_frames > 0 && window.timed_frames > 0 && scope_runs > 0.0 {
            let plain_ms = window.plain_render_ms / f64::from(window.plain_frames);
            let timed_ms = window.timed_render_ms / f64::from(window.timed_frames);
            let estimate = (timed_ms - plain_ms) * f64::from(window.timed_frames) / scope_runs;
            if self.scope_estimates.len() == SCOPE_ESTIMATES {
                self.scope_estimates.pop_front();
            }
            self.scope_estimates.push_back(estimate);
        }
        let scope_ms = median(&self.scope_estimates).max(0.0);

        let timed_frames = self.timed_frames as f64;
        for (name, runs, read_ms) in scopes {
            self.overhead_ms += runs * scope_ms;
            let pass = self.passes.entry(name).or_default();
            pass.net_ms += (read_ms - runs * scope_ms / 2.0).max(0.0);
            if name != WHOLE_RENDER_SCOPE {
                entries.push(total(
                    TimingKind::RendererPass,
                    name,
                    pass.net_ms,
                    timed_frames,
                ));
            }
        }
        // Averaged over every frame: what looking adds to the renderer's
        // time per frame.
        entries.push(total(
            TimingKind::RendererOverhead,
            OVERHEAD_NAME,
            self.overhead_ms,
            plain_frames + timed_frames,
        ));
    }
}

/// The middle value (the upper of the two for an even count), or 0 for none.
fn median(values: &VecDeque<f64>) -> f64 {
    let mut sorted: Vec<f64> = values.iter().copied().collect();
    sorted.sort_by(f64::total_cmp);
    sorted.get(sorted.len() / 2).copied().unwrap_or(0.0)
}

fn total(kind: TimingKind, name: &str, total_ms: f64, count: f64) -> TimingTotal {
    TimingTotal {
        kind,
        name: name.to_string(),
        total_ms,
        count,
    }
}
