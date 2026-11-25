//! What the render worker tells the page's Stats panel.
//!
//! awsm-renderer-stats does the measuring and the formatting: frame windows,
//! smoothing, the frame budget, the display group (from the page's own frame
//! gaps), the renderer's CPU scopes and GPU passes, and the page group (from
//! the page's own work times). Its `StatsCollector` runs here, on the thread
//! that renders, and posts a report about once a second; the page's
//! `StatsPanel` only shows it, and sends back which groups are open.
//!
//! This file adds what is particular to this project: the names of the
//! render frame's phases, and the physics group, built from what the physics
//! thread keeps in shared memory ([`BodyMotion`]'s step counters and step
//! phases, the [`TaskPool`]'s per-executor task time) plus the dropped-ball
//! count. The physics thread times its step phases only while that group is
//! open ([`DETAIL_PHYSICS`], which the render loop sets from
//! [`StatsCollector::is_open`]).

use awsm_renderer_stats::{
    duration_row, step_headline, AppSection, NamedMeans, SectionSpec, StatRow, StatsCollector,
    StatsConfig, Window,
};

use crate::physics_tasks::TaskPool;
use crate::protocol::{BodyMotion, STEP_PHASES};

/// The phases of one render frame, in the order they run; the Stats panel
/// lists them by these names ([`PHASE_NAMES`]).
#[derive(Clone, Copy)]
pub enum Phase {
    /// Everything up to the transforms being written: waking physics, the
    /// camera ring, sampling + interpolating the published steps, minting
    /// dropped balls.
    ReadApp,
    /// The scene's clips + texture UV flows (`update_animations`) and its
    /// particles (`ParticleDriver::tick`). Runs inside `ReadApp`, before the
    /// ball writes.
    Animate,
    SetCamera,
    UpdateTransforms,
    /// `AwsmRenderer::render`: encoding every pass and submitting.
    Render,
}

const PHASE_NAMES: [&str; 5] = [
    "read app state + interpolate",
    "animations + particles",
    "set camera",
    "update transforms",
    "render (encode + submit)",
];

/// The physics group's id (the page stores whether it is open under it).
pub const PHYSICS: &str = "physics";

/// The physics group's label, which names the pool's size once known.
fn physics_label(pool_workers: Option<u32>) -> String {
    match pool_workers {
        Some(1) => "physics (Box3D, physics worker + 1 pool thread)".to_string(),
        Some(n) => format!("physics (Box3D, physics worker + {n} pool threads)"),
        None => "physics (Box3D, physics worker)".to_string(),
    }
}

/// The Stats panel's configuration: the render frame's phases, and the
/// physics group (green lane) between the renderer and the page.
pub fn config() -> StatsConfig {
    StatsConfig {
        phases: PHASE_NAMES.map(String::from).to_vec(),
        render_phase: Some(Phase::Render as usize),
        sections: vec![SectionSpec::new(PHYSICS, physics_label(None), Some("sim"))],
        // The renderer draws on every refresh: no pacing row.
        pacing_row: false,
        ..StatsConfig::default()
    }
}

/// The shared counters at the previous report.
struct Sample {
    step: u32,
    step_work_us: u32,
    /// Per-executor accumulated task CPU µs (index 0 = the physics worker).
    task_us: Vec<u32>,
    step_phases: Option<([u32; STEP_PHASES.len()], u32)>,
}

impl Sample {
    fn read(motion: &BodyMotion, pool: Option<(usize, u32)>) -> Self {
        Self {
            step: motion.latest_step(),
            step_work_us: motion.step_work_us(),
            task_us: match pool {
                // SAFETY: a leaked-for-the-session block in the shared
                // memory (see `TaskPool::leak_new`); only atomics are read.
                Some((addr, workers)) => {
                    unsafe { &*(addr as *const TaskPool) }.task_us_counts(workers as usize + 1)
                }
                None => Vec::new(),
            },
            step_phases: motion.step_phases(),
        }
    }
}

/// The physics group's state between reports.
#[derive(Default)]
pub struct PhysicsStats {
    baseline: Option<Sample>,
    phases: NamedMeans,
}

impl PhysicsStats {
    /// The physics group for one window: its CPU time per fixed step
    /// (including the time the physics worker waits on the pool inside a
    /// step), and its time per drawn frame as the budget lane. Its rows:
    /// steps per second, the frames drawn before physics had published the
    /// moment they show (`late`), the step's phases (while open), how the
    /// solver's task time split between the physics worker and the pool,
    /// and the balls dropped.
    pub fn section(
        &mut self,
        stats: &mut StatsCollector,
        window: &Window,
        motion: &BodyMotion,
        balls: usize,
    ) -> AppSection {
        let pool = motion.pool();
        let spec = SectionSpec::new(PHYSICS, physics_label(pool.map(|p| p.1)), Some("sim"));
        let now = Sample::read(motion, pool);
        let Some(previous) = self.baseline.replace(now) else {
            return spec.group().headline("—");
        };
        let now = self.baseline.as_ref().expect("just set");

        // Wrapping totals: the change since the last report.
        let steps = now.step.wrapping_sub(previous.step);
        let work_ms = f64::from(now.step_work_us.wrapping_sub(previous.step_work_us)) / 1000.0;
        let steps_per_s = stats.smooth("steps", f64::from(steps) / window.seconds);
        let per_step = (steps > 0)
            .then(|| stats.smooth("step", work_ms / f64::from(steps)))
            .or(stats.smoothed("step"));
        // The same time per frame drawn: what the physics thread uses of a
        // frame's budget. The pool threads get no lane: they run in parallel
        // with the physics worker, inside the step it times.
        let per_frame = (window.frames > 0)
            .then(|| stats.smooth("frame", work_ms / f64::from(window.frames)))
            .or(stats.smoothed("frame"));

        let mut rows = vec![
            StatRow::new("steps", format!("{steps_per_s:.0} steps/s")),
            window.late.row("late frames"),
        ];

        // The step phases: each one's mean per step over the window. Phases
        // that did not advance are no longer being timed (the group was just
        // opened), so they drop out rather than show a stale figure.
        let phase_steps = match (previous.step_phases, now.step_phases) {
            (Some((_, s0)), Some((_, s1))) => s1.wrapping_sub(s0),
            _ => 0,
        };
        match (previous.step_phases, now.step_phases) {
            (Some((ns0, _)), Some((ns1, _))) if stats.is_open(PHYSICS) && phase_steps > 0 => {
                self.phases
                    .update(STEP_PHASES.iter().enumerate().map(|(i, name)| {
                        let ms =
                            f64::from(ns1[i].wrapping_sub(ns0[i])) / f64::from(phase_steps) / 1e6;
                        (name.to_string(), ms)
                    }));
            }
            _ => self.phases.clear(),
        }
        rows.extend(
            self.phases
                .entries()
                .iter()
                .map(|(name, ms)| duration_row(name.clone(), *ms, "step")),
        );

        // The solver's task time as a TRUE work share: each executor sums the
        // CPU µs its claimed tasks took (index 0 is the physics worker
        // help-executing in `finishTask`), so a 500 µs task counts 100× a 5 µs
        // one. The per-worker split is scheduler randomness; only pool vs.
        // physics worker is shown.
        if let Some((_, workers)) = pool {
            if now.task_us.len() > 1 && now.task_us.len() == previous.task_us.len() {
                let deltas: Vec<u64> = now
                    .task_us
                    .iter()
                    .zip(&previous.task_us)
                    .map(|(&now, &then)| u64::from(now.wrapping_sub(then)))
                    .collect();
                let total: u64 = deltas.iter().sum();
                let (own, pooled) =
                    match (deltas[1..].iter().sum::<u64>() * 100 + total / 2).checked_div(total) {
                        Some(pct) => (
                            format!("{}% of task time", 100 - pct),
                            format!("{pct}% of task time"),
                        ),
                        None => ("idle".to_string(), "idle".to_string()),
                    };
                rows.push(StatRow::new("solver tasks: physics worker", own));
                rows.push(StatRow::new(
                    format!("solver tasks: {workers} pool threads"),
                    pooled,
                ));
            }
        }
        rows.push(StatRow::new("balls dropped", balls.to_string()));

        spec.group()
            .headline(step_headline(per_step))
            .bad(window.late.bad)
            .budget_ms(per_frame)
            .rows(rows)
    }
}
