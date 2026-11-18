//! The Stats panel (top right) and the Stats button (bottom left).
//!
//! One group per thread, so every number sits under the thread that spent
//! it, and every number says its unit and what it is per:
//!
//! - `frame budget`: the refresh period, and each thread's and the GPU's
//!   time per drawn frame against it.
//! - `display`: the refresh rate of the browser's frame clock (measured here,
//!   see `refresh`), the frames the renderer presented per second, and the
//!   refreshes that went by with no new frame.
//! - `GPU`: the GPU's time per frame (renderer timestamp queries; `n/a`
//!   without them). Its rows are the passes; on GPUs that run passes
//!   overlapped (Apple's) they add up to more than the frame, which is
//!   expected, not a bug.
//! - `renderer (render worker)`: the render worker's CPU time per frame;
//!   expanded, our phases and awsm-renderer's CPU scopes inside `render()`.
//! - `physics (Box3D, physics worker + N pool threads)`: the physics
//!   worker's CPU time per fixed step (including the time it waits on the
//!   pool inside a step), steps per second, frames drawn before physics had
//!   published the moment they show ("late"), and how the solver's task time
//!   split between the physics worker and the pool.
//! - `page (main thread)`: this thread's own per-frame work.
//!
//! A group's header always shows its headline. Expanding a group shows its
//! breakdown, and only then is the breakdown measured: main sets the group's
//! `DETAIL_*` bit in [`BodyMotion::detail`] and the workers time their phases
//! and passes only while it is set. A collapsed group costs nothing.
//!
//! A 1 Hz timer samples the counters the workers keep in shared memory (plus
//! the renderer's [`RenderMsg::Timings`] totals). Every figure is the change in
//! a total divided by the change in what it is per (frames, steps, or REAL
//! seconds: the timer wobbles and is throttled in a background tab), then
//! EMA-smoothed.
//!
//! [`RenderMsg::Timings`]: crate::protocol::RenderMsg::Timings

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

use dominator::{clone, html, Dom};
use futures_signals::signal::{Mutable, Signal, SignalExt};
use futures_signals::signal_vec::{MutableVec, SignalVecExt};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use crate::main_thread::{coarse_pointer, store_bool, stored_bool, STATS_STORAGE_KEY};
use crate::physics_tasks::TaskPool;
use crate::protocol::{
    BallMotions, BodyMotion, TimingKind, TimingTotal, DETAIL_GPU, DETAIL_PHYSICS, DETAIL_RENDERER,
    STEP_PHASES,
};

/// Smoothing weight of each new sample: a real change shows within 2–3
/// samples, while the readout holds steady against noise.
const EMA_ALPHA: f64 = 0.35;
/// A sample window is only trusted between these lengths (ms). Shorter: two
/// timer callbacks ran almost together. Longer: the tab was hidden or
/// throttled, and the window measures the stall.
const MIN_WINDOW_MS: f64 = 200.0;
const MAX_WINDOW_MS: f64 = 3000.0;
/// The late-frames line (and the physics header) turns red when physics was
/// late in this many windows in a row: a condition, not a one-off.
const LATE_WINDOWS: u32 = 3;
/// Frames presented after the renderer's `Ready` before anything is counted:
/// boot (pipeline compiles, cold caches, the wasm engine tiering up) reads
/// 2–3× the settled cost. Counted in frames, so a stalled renderer keeps
/// waiting instead of "warming up" through the stall.
const WARMUP_FRAMES: u32 = 60;
/// The renderer phase whose time its scope rows break down.
const RENDER_PHASE: &str = "render (encode + submit)";
/// Scopes under this mean (ms per frame) are folded into one line.
const FOLD_BELOW_MS: f64 = 0.001;
/// The renderer's scope kinds, in the panel's order, with their labels.
const SCOPE_GROUPS: [(TimingKind, &str); 4] = [
    (TimingKind::RendererPrep, "prepare"),
    (TimingKind::RendererPass, "passes"),
    (TimingKind::RendererSubmit, "submit"),
    (TimingKind::RendererHook, "hooks"),
];

/// The groups, top to bottom.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Group {
    Budget,
    Display,
    Gpu,
    Renderer,
    Physics,
    Page,
}

impl Group {
    const ALL: [Group; 6] = [
        Group::Budget,
        Group::Display,
        Group::Gpu,
        Group::Renderer,
        Group::Physics,
        Group::Page,
    ];

    /// Storage / element id.
    fn id(self) -> &'static str {
        match self {
            Group::Budget => "budget",
            Group::Display => "display",
            Group::Gpu => "gpu",
            Group::Renderer => "renderer",
            Group::Physics => "physics",
            Group::Page => "page",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Group::Budget => "frame budget",
            Group::Display => "display",
            Group::Gpu => "GPU",
            Group::Renderer => "renderer (render worker)",
            Group::Physics => "physics (Box3D, physics worker)",
            Group::Page => "page (main thread)",
        }
    }

    /// The bit that asks the workers to measure this group's breakdown.
    fn detail(self) -> u32 {
        match self {
            Group::Gpu => DETAIL_GPU,
            Group::Renderer => DETAIL_RENDERER,
            Group::Physics => DETAIL_PHYSICS,
            Group::Budget | Group::Display | Group::Page => 0,
        }
    }
}

/// The `localStorage` key of a group's open/closed state.
fn open_key(group: Group) -> String {
    format!("{STATS_STORAGE_KEY}_open_{}", group.id())
}

/// Every group's open/closed key, for `?reset`.
pub fn open_keys() -> impl Iterator<Item = String> {
    Group::ALL.into_iter().map(open_key)
}

/// A duration with its unit: microseconds below a tenth of a millisecond.
fn duration(ms: f64) -> String {
    if ms < 0.09995 {
        let us = ms * 1000.0;
        if us < 9.95 {
            format!("{us:.1} µs")
        } else {
            format!("{us:.0} µs")
        }
    } else if ms < 9.995 {
        format!("{ms:.2} ms")
    } else {
        format!("{ms:.1} ms")
    }
}

fn per_frame(ms: f64) -> String {
    format!("{} / frame", duration(ms))
}

fn per_step(ms: f64) -> String {
    format!("{} / step", duration(ms))
}

/// Milliseconds with one decimal, for the frame budget: `<0.1` for a
/// nonzero time that would print as 0.0.
fn ms_1dp(ms: f64) -> String {
    if ms > 0.0 && ms < 0.05 {
        "<0.1".to_string()
    } else {
        format!("{ms:.1}")
    }
}

/// A frame budget row: `7.2 / 16.7 ms (43%)`.
fn budget_share(used_ms: f64, budget_ms: f64) -> String {
    let pct = used_ms / budget_ms * 100.0;
    let pct = if pct > 0.0 && pct < 1.0 {
        "<1%".to_string()
    } else {
        format!("{pct:.0}%")
    };
    format!("{} / {budget_ms:.1} ms ({pct})", ms_1dp(used_ms))
}

/// The physics group's label, which names the pool's size once known.
fn physics_label(pool: Option<(usize, u32)>) -> String {
    match pool {
        Some((_, 1)) => "physics (Box3D, physics worker + 1 pool thread)".to_string(),
        Some((_, n)) => format!("physics (Box3D, physics worker + {n} pool threads)"),
        None => Group::Physics.label().to_string(),
    }
}

fn smooth(ema: Option<f64>, value: f64) -> f64 {
    match ema {
        Some(e) => e + (value - e) * EMA_ALPHA,
        None => value,
    }
}

/// One row of a group: label left, value right. `depth` indents it under
/// the row above (a breakdown of it), up to two levels; `bad` turns it red.
#[derive(Clone, PartialEq)]
struct Row {
    label: String,
    value: String,
    depth: u8,
    bad: bool,
}

fn row(label: impl Into<String>, value: impl Into<String>) -> Row {
    Row {
        label: label.into(),
        value: value.into(),
        depth: 0,
        bad: false,
    }
}

impl Row {
    fn nested(mut self) -> Self {
        self.depth = 1;
        self
    }

    fn nested_2(mut self) -> Self {
        self.depth = 2;
        self
    }

    fn bad(mut self, bad: bool) -> Self {
        self.bad = bad;
        self
    }
}

struct GroupView {
    group: Group,
    label: Mutable<String>,
    open: Mutable<bool>,
    headline: Mutable<String>,
    bad: Mutable<bool>,
    rows: MutableVec<Row>,
}

/// The panel's state: whether it is shown, and per group whether it is
/// expanded and what it shows.
pub struct Panel {
    visible: Mutable<bool>,
    groups: Vec<GroupView>,
    /// The `DETAIL_*` bits of the expanded groups, 0 while the panel is hidden.
    detail: Mutable<u32>,
}

impl Panel {
    /// Hidden by default: on a phone the panel covers a lot of the screen, so
    /// a stored "open" is only honoured on a desktop. Every group starts
    /// collapsed unless the user expanded it before.
    pub fn new() -> Rc<Self> {
        let window = web_sys::window();
        let stored = |key: &str| {
            window
                .as_ref()
                .and_then(|w| stored_bool(w, key))
                .unwrap_or(false)
        };
        let visible = window
            .as_ref()
            .is_some_and(|w| !coarse_pointer(w) && stored(STATS_STORAGE_KEY));
        let groups = Group::ALL
            .into_iter()
            .map(|group| GroupView {
                group,
                label: Mutable::new(group.label().to_string()),
                open: Mutable::new(stored(&open_key(group))),
                headline: Mutable::new(String::new()),
                bad: Mutable::new(false),
                rows: MutableVec::new(),
            })
            .collect();
        let panel = Rc::new(Panel {
            visible: Mutable::new(visible),
            groups,
            detail: Mutable::new(0),
        });
        panel.update_detail();
        panel.placeholder("starting…");
        panel
    }

    fn view(&self, group: Group) -> &GroupView {
        &self.groups[group as usize]
    }

    fn update_detail(&self) {
        let bits = if self.visible.get() {
            self.groups
                .iter()
                .filter(|g| g.open.get())
                .fold(0, |bits, g| bits | g.group.detail())
        } else {
            0
        };
        self.detail.set_neq(bits);
    }

    /// The `DETAIL_*` bits the workers should measure right now.
    pub fn detail(&self) -> u32 {
        self.detail.get()
    }

    pub fn detail_signal(&self) -> impl Signal<Item = u32> {
        self.detail.signal()
    }

    fn set_label(&self, group: Group, label: String) {
        self.view(group).label.set_neq(label);
    }

    fn set(&self, group: Group, headline: impl Into<String>, rows: Vec<Row>, bad: bool) {
        let view = self.view(group);
        view.headline.set_neq(headline.into());
        view.bad.set_neq(bad);
        let mut current = view.rows.lock_mut();
        if current[..] != rows[..] {
            current.replace_cloned(rows);
        }
    }

    /// What every group says until there are numbers.
    fn placeholder(&self, text: &str) {
        for group in Group::ALL {
            let headline = if group == Group::Display { "—" } else { text };
            self.set(group, headline, Vec::new(), false);
        }
    }

    /// The panel: a `what do these mean?` button that calls `on_help`, then
    /// per group a header button (label left, headline right) and the rows
    /// it shows while expanded.
    pub fn dom(self: &Rc<Self>, on_help: impl Fn() + 'static) -> Dom {
        let panel = self.clone();
        html!("div", {
            .class("stats")
            .visible_signal(self.visible.signal())
            .child(html!("button", {
                .attr("type", "button")
                .class("stats-help")
                .text("what do these mean?")
                .event(move |_: dominator::events::Click| on_help())
            }))
            .children(self.groups.iter().enumerate().flat_map(|(index, view)| {
                let rows_id = format!("stats-rows-{}", view.group.id());
                [
                    html!("button", {
                        .attr("type", "button")
                        .class("stats-group")
                        .class_signal("stats-bad", view.bad.signal())
                        .attr("aria-controls", &rows_id)
                        .attr_signal("aria-expanded", view.open.signal().map(|open| if open { "true" } else { "false" }))
                        .child(html!("span", {
                            .text_signal(view.open.signal().map(|open| if open { "▾ " } else { "▸ " }))
                            .text_signal(view.label.signal_cloned())
                        }))
                        .text(" ") // for a screen reader; the layout ignores it
                        .child(html!("span", {
                            .text_signal(view.headline.signal_cloned())
                        }))
                        .event(clone!(panel => move |_: dominator::events::Click| {
                            let view = &panel.groups[index];
                            let open = !view.open.get();
                            view.open.set(open);
                            if let Some(w) = web_sys::window() {
                                store_bool(&w, &open_key(view.group), open);
                            }
                            panel.update_detail();
                        }))
                    }),
                    html!("div", {
                        .class("stats-rows")
                        .attr("id", &rows_id)
                        .prop_signal("hidden", view.open.signal().map(|open| !open))
                        .children_signal_vec(view.rows.signal_vec_cloned().map(|row| {
                            html!("div", {
                                .class("stats-row")
                                .apply_if(row.depth == 1, |dom| dom.class("stats-nested"))
                                .apply_if(row.depth == 2, |dom| dom.class("stats-nested-2"))
                                .apply_if(row.bad, |dom| dom.class("stats-bad"))
                                .child(html!("span", { .text(&row.label) }))
                                .child(html!("span", { .text(&row.value) }))
                            })
                        }))
                    }),
                ]
            }).collect::<Vec<_>>())
        })
    }

    /// The bottom-left "Stats" chip: shows/hides the panel, persisted.
    pub fn button(self: &Rc<Self>) -> Dom {
        let panel = self.clone();
        html!("button", {
            .attr("type", "button")
            .class("stats-btn")
            .class_signal("active", self.visible.signal())
            .text("Stats")
            .event(move |_: dominator::events::Click| {
                let next = !panel.visible.get();
                panel.visible.set_neq(next);
                if let Some(w) = web_sys::window() {
                    store_bool(&w, STATS_STORAGE_KEY, next);
                }
                panel.update_detail();
            })
        })
    }
}

/// Shared-memory addresses the panel samples (filled in as the workers come
/// up): the player's [`BodyMotion`] (frame tick, step count, timing counters,
/// the detail bits), the [`BallMotions`] block (dropped-ball count), and the
/// task pool + its worker count. All are leaked-for-the-session blocks, so
/// the raw addresses are stable.
#[derive(Default, Clone, Copy)]
pub struct StatsRefs {
    pub motion: Option<usize>,
    pub balls: Option<usize>,
    pub pool: Option<(usize, u32)>,
}

/// This thread's own per-frame work, summed between samples: its camera
/// frame loop, and the sound cues it plays for physics.
#[derive(Default)]
pub struct PageWork {
    frame_ms: Cell<f64>,
    cues_ms: Cell<f64>,
    frames: Cell<u32>,
}

impl PageWork {
    /// One animation frame of the main thread, which took `ms`.
    pub fn frame(&self, ms: f64) {
        self.frame_ms.set(self.frame_ms.get() + ms);
        self.frames.set(self.frames.get() + 1);
    }

    /// Time spent handling sound cues.
    pub fn cues(&self, ms: f64) {
        self.cues_ms.set(self.cues_ms.get() + ms);
    }

    fn take(&self) -> (f64, f64, u32) {
        (
            self.frame_ms.take(),
            self.cues_ms.take(),
            self.frames.take(),
        )
    }
}

/// One sample of the shared counters, plus WHEN it was read
/// (`performance.now()` ms).
struct Sample {
    t: f64,
    frame: u32,
    step: u32,
    frame_work_us: u32,
    step_work_us: u32,
    /// Renderer GPU frame-time totals: (µs, samples), wrapping.
    gpu: (u32, u32),
    /// Accumulated time between presented frames (µs, wrapping).
    frame_span_us: u32,
    /// Frames drawn with physics late (wrapping).
    app_late: u32,
    /// Per-executor accumulated task CPU µs (index 0 = the physics worker).
    task_us: Vec<u32>,
    /// The step-phase totals (`BodyMotion::step_phases`).
    step_phases: Option<([u32; STEP_PHASES.len()], u32)>,
}

/// Everything the panel shows, smoothed across samples.
#[derive(Default)]
struct Figures {
    fps: Option<f64>,
    steps: Option<f64>,
    frame_work: Option<f64>,
    step_work: Option<f64>,
    /// The physics worker's step time per drawn frame.
    physics_per_frame: Option<f64>,
    gpu: Option<f64>,
    /// Trusted windows in a row with no GPU sample: after a few, the device
    /// evidently has no `timestamp-query`.
    gpu_empty_windows: u32,
    page: Option<f64>,
    page_frame: Option<f64>,
    page_cues: Option<f64>,
    late_total: u32,
    late_windows: u32,
    late_rate: f64,
    /// The solver's task-time split this window: `Some(pool %)`, or `None`
    /// while it was idle.
    pool_share: Option<Option<u64>>,
    step_phases: [Option<f64>; STEP_PHASES.len()],
    /// The renderer's breakdowns: mean ms per frame by entry.
    timings: HashMap<(TimingKind, String), f64>,
}

impl Figures {
    fn timing(&self, kind: TimingKind, name: &str) -> Option<f64> {
        self.timings.get(&(kind, name.to_string())).copied()
    }
}

/// Starts the 1 Hz sampler. `render_timings` holds the latest
/// [`RenderMsg::Timings`] totals; `render_ready` turns true once the renderer
/// is drawing for real.
///
/// [`RenderMsg::Timings`]: crate::protocol::RenderMsg::Timings
pub fn install_stats(
    window: &web_sys::Window,
    panel: Rc<Panel>,
    refs: Rc<RefCell<StatsRefs>>,
    render_ready: Rc<Cell<bool>>,
    render_timings: Rc<RefCell<Vec<TimingTotal>>>,
    page: Rc<PageWork>,
) -> Result<(), JsValue> {
    let perf = window
        .performance()
        .ok_or_else(|| JsValue::from_str("no performance"))?;
    let refresh = crate::refresh::install(window)?;
    let mut missed = crate::refresh::Missed::default();

    // Tell the workers what to measure as soon as a group is toggled (the
    // sampler below repeats it, for the workers that come up later).
    wasm_bindgen_futures::spawn_local(panel.detail_signal().for_each(clone!(refs => move |bits| {
        if let Some(addr) = refs.borrow().motion {
            // SAFETY: a leaked-for-the-session block in the shared memory.
            unsafe { &*(addr as *const BodyMotion) }.set_detail(bits);
        }
        async {}
    })));

    // The previous sample. `None` forces a re-seed: at startup, and whenever
    // the tab becomes visible again (a hidden tab runs no frames).
    let baseline: Rc<RefCell<Option<Sample>>> = Rc::new(RefCell::new(None));
    {
        let baseline = baseline.clone();
        let vis = Closure::<dyn FnMut()>::new(move || {
            *baseline.borrow_mut() = None;
        });
        window
            .document()
            .ok_or_else(|| JsValue::from_str("no document"))?
            .add_event_listener_with_callback("visibilitychange", vis.as_ref().unchecked_ref())?;
        vis.forget();
    }

    let mut figures = Figures::default();
    // The renderer's totals at the previous sample, by entry.
    let mut timing_baseline: HashMap<(TimingKind, String), (f64, f64)> = HashMap::new();
    // The frame tick when `Ready` was first seen; counting starts
    // `WARMUP_FRAMES` presented frames after it.
    let mut warm_frame: Option<u32> = None;

    let tick = Closure::<dyn FnMut()>::new(move || {
        let refs = *refs.borrow();
        let now = perf.now();
        // SAFETY (all blocks): leaked-for-the-session allocations in the
        // shared `WebAssembly.Memory`; we only touch atomics.
        let motion = refs
            .motion
            .map(|addr| unsafe { &*(addr as *const BodyMotion) });
        if let Some(motion) = motion {
            motion.set_detail(panel.detail());
        }
        panel.set_label(Group::Physics, physics_label(refs.pool));
        let page_work = page.take();
        let timings: HashMap<(TimingKind, String), (f64, f64)> = render_timings
            .borrow()
            .iter()
            .map(|t| ((t.kind, t.name.clone()), (t.total_ms, t.count)))
            .collect();
        let previous_timings = std::mem::replace(&mut timing_baseline, timings);

        let Some(motion) = motion else {
            figures = Figures::default();
            missed.reset();
            panel.placeholder("starting…");
            return;
        };
        // Read the span AFTER the frame tick: render adds a frame's span
        // before bumping its tick, so the span covers the frames counted.
        let frame = motion.frame_tick();
        let sample = Sample {
            t: now,
            frame,
            step: motion.latest_step(),
            frame_work_us: motion.frame_work_us(),
            step_work_us: motion.step_work_us(),
            gpu: motion.gpu_totals(),
            frame_span_us: motion.frame_span_us(),
            app_late: motion.app_late(),
            task_us: match refs.pool {
                Some((addr, count)) => {
                    unsafe { &*(addr as *const TaskPool) }.task_us_counts(count as usize + 1)
                }
                None => Vec::new(),
            },
            step_phases: motion.step_phases(),
        };
        let previous = baseline.borrow_mut().replace(sample);
        let stored = baseline.borrow();
        let Some(current) = stored.as_ref() else {
            return;
        };

        // Nothing is counted until the renderer is drawing for real, and then
        // only after WARMUP_FRAMES more frames. The baseline keeps moving, so
        // the first window shown starts exactly where the warm-up ends.
        if !render_ready.get() {
            return; // still "starting…"
        }
        let base = *warm_frame.get_or_insert(frame);
        if frame.wrapping_sub(base) < WARMUP_FRAMES {
            panel.placeholder("warming up…");
            return;
        }
        // No trustworthy window (the first sample, or a jittered or throttled
        // gap): the baseline has been re-seeded above; keep the last figures.
        let Some(previous) =
            previous.filter(|p| (MIN_WINDOW_MS..MAX_WINDOW_MS).contains(&(now - p.t)))
        else {
            return;
        };
        sample_window(
            &mut figures,
            &previous,
            current,
            &previous_timings,
            &timing_baseline,
            page_work,
        );
        let refresh = refresh.borrow();
        missed.window(
            current.frame.wrapping_sub(previous.frame),
            current.frame_span_us.wrapping_sub(previous.frame_span_us) as f64 / 1000.0,
            now - previous.t,
            refresh.period_ms(),
        );
        let balls = refs
            .balls
            .map(|addr| unsafe { &*(addr as *const BallMotions) }.count());
        show(&panel, &figures, &refresh, &missed, balls, refs.pool);
    });
    window.set_interval_with_callback_and_timeout_and_arguments_0(
        tick.as_ref().unchecked_ref(),
        1000,
    )?;
    tick.forget();
    Ok(())
}

/// Folds one trusted window (`previous` → `current`) into the figures.
fn sample_window(
    figures: &mut Figures,
    previous: &Sample,
    current: &Sample,
    previous_timings: &HashMap<(TimingKind, String), (f64, f64)>,
    timings: &HashMap<(TimingKind, String), (f64, f64)>,
    (page_frame_ms, page_cues_ms, page_frames): (f64, f64, u32),
) {
    let seconds = (current.t - previous.t) / 1000.0;
    let frames = current.frame.wrapping_sub(previous.frame);
    let steps = current.step.wrapping_sub(previous.step);
    figures.fps = Some(smooth(figures.fps, frames as f64 / seconds));
    figures.steps = Some(smooth(figures.steps, steps as f64 / seconds));
    // Average CPU ms per frame / per step over the window. With nothing to
    // divide by, the previous value stays.
    if frames > 0 {
        let us = current.frame_work_us.wrapping_sub(previous.frame_work_us);
        figures.frame_work = Some(smooth(
            figures.frame_work,
            us as f64 / frames as f64 / 1000.0,
        ));
        let us = current.step_work_us.wrapping_sub(previous.step_work_us);
        figures.physics_per_frame = Some(smooth(
            figures.physics_per_frame,
            us as f64 / frames as f64 / 1000.0,
        ));
    }
    if steps > 0 {
        let us = current.step_work_us.wrapping_sub(previous.step_work_us);
        figures.step_work = Some(smooth(figures.step_work, us as f64 / steps as f64 / 1000.0));
    }
    if page_frames > 0 {
        let per = |ms: f64| ms / page_frames as f64;
        figures.page = Some(smooth(figures.page, per(page_frame_ms + page_cues_ms)));
        figures.page_frame = Some(smooth(figures.page_frame, per(page_frame_ms)));
        figures.page_cues = Some(smooth(figures.page_cues, per(page_cues_ms)));
    }
    // Average GPU ms per SAMPLED frame: the readback skips frames while one
    // is in flight, so this divides by GPU samples, not by frames.
    let gpu_samples = current.gpu.1.wrapping_sub(previous.gpu.1);
    if gpu_samples > 0 {
        figures.gpu_empty_windows = 0;
        let us = current.gpu.0.wrapping_sub(previous.gpu.0);
        figures.gpu = Some(smooth(figures.gpu, us as f64 / gpu_samples as f64 / 1000.0));
    } else {
        figures.gpu_empty_windows = figures.gpu_empty_windows.saturating_add(1);
    }

    let late = current.app_late.wrapping_sub(previous.app_late);
    figures.late_total = figures.late_total.wrapping_add(late);
    figures.late_windows = if late > 0 {
        figures.late_windows + 1
    } else {
        0
    };
    figures.late_rate = late as f64 / seconds;

    // The solver's task time as a TRUE work share: each executor sums the
    // CPU µs its claimed tasks took (index 0 is the physics worker
    // help-executing in `finishTask`), so a 500 µs task counts 100× a 5 µs
    // one. The per-worker split is scheduler randomness; only pool vs.
    // physics worker is shown.
    figures.pool_share =
        (current.task_us.len() > 1 && current.task_us.len() == previous.task_us.len()).then(|| {
            let deltas: Vec<u64> = current
                .task_us
                .iter()
                .zip(&previous.task_us)
                .map(|(&now, &then)| now.wrapping_sub(then) as u64)
                .collect();
            let total: u64 = deltas.iter().sum();
            (deltas[1..].iter().sum::<u64>() * 100 + total / 2).checked_div(total)
        });

    // The step phases: each one's mean over the window. Phases that did not
    // advance are no longer being timed (the group was collapsed), so they
    // drop out rather than show a stale figure.
    let phase_steps = match (previous.step_phases, current.step_phases) {
        (Some((_, s0)), Some((_, s1))) => s1.wrapping_sub(s0),
        _ => 0,
    };
    match (previous.step_phases, current.step_phases) {
        (Some((ns0, _)), Some((ns1, _))) if phase_steps > 0 => {
            for (i, ema) in figures.step_phases.iter_mut().enumerate() {
                let ms = ns1[i].wrapping_sub(ns0[i]) as f64 / phase_steps as f64 / 1e6;
                *ema = Some(smooth(*ema, ms));
            }
        }
        _ => figures.step_phases = Default::default(),
    }

    // The renderer's breakdowns, the same way. A pass or scope whose total
    // did not move did not run (switched off), so it drops out too.
    let mut means = HashMap::new();
    for (key, &(total_ms, count)) in timings {
        let Some(&(total0, count0)) = previous_timings.get(key) else {
            continue;
        };
        if count > count0 && (key.0 == TimingKind::RendererPhase || total_ms > total0) {
            let ms = (total_ms - total0) / (count - count0);
            means.insert(key.clone(), smooth(figures.timings.get(key).copied(), ms));
        }
    }
    figures.timings = means;
}

/// Writes the figures into the panel.
fn show(
    panel: &Panel,
    figures: &Figures,
    refresh: &crate::refresh::Refresh,
    missed: &crate::refresh::Missed,
    balls: Option<usize>,
    pool: Option<(usize, u32)>,
) {
    show_budget(panel, figures, refresh.period_ms(), pool);

    let fps = figures.fps.unwrap_or(0.0);
    let hz = refresh.line();
    panel.set(
        Group::Display,
        format!("{hz} · {fps:.0} fps"),
        vec![
            row("refresh rate", hz),
            row("frames presented", format!("{fps:.0} / s")),
            row("missed frames", missed.line()),
        ],
        false,
    );

    // One row per kind of entry, by name; passes slowest first.
    let entries = |kind: TimingKind| {
        let mut entries: Vec<(&str, f64)> = figures
            .timings
            .iter()
            .filter(|((k, _), _)| *k == kind)
            .map(|((_, name), &ms)| (name.as_str(), ms))
            .collect();
        entries.sort_by(|a, b| b.1.total_cmp(&a.1));
        entries
    };

    let gpu = match figures.gpu {
        Some(ms) => per_frame(ms),
        None if figures.gpu_empty_windows >= 3 => "n/a".to_string(),
        None => "—".to_string(),
    };
    // On GPUs that overlap passes (Apple's), these add up to more than the
    // frame total above; that is expected.
    let gpu_rows = entries(TimingKind::GpuPass)
        .into_iter()
        .map(|(name, ms)| row(name, per_frame(ms)))
        .collect();
    panel.set(Group::Gpu, gpu, gpu_rows, false);

    // Our phases in the order they run, with awsm-renderer's scopes inside
    // render() grouped by kind under the render phase.
    let mut renderer_rows = Vec::new();
    for name in crate::render_profile::PHASE_NAMES {
        let Some(ms) = figures.timing(TimingKind::RendererPhase, name) else {
            continue;
        };
        renderer_rows.push(row(name, per_frame(ms)));
        if name != RENDER_PHASE {
            continue;
        }
        let mut in_scopes = None;
        for (kind, label) in SCOPE_GROUPS {
            let scopes = entries(kind);
            if scopes.is_empty() {
                continue;
            }
            let sum: f64 = scopes.iter().map(|(_, ms)| ms).sum();
            *in_scopes.get_or_insert(0.0) += sum;
            renderer_rows.push(row(label, per_frame(sum)).nested());
            let (shown, folded): (Vec<_>, Vec<_>) =
                scopes.into_iter().partition(|&(_, ms)| ms >= FOLD_BELOW_MS);
            for (scope, scope_ms) in shown {
                renderer_rows.push(row(scope, per_frame(scope_ms)).nested_2());
            }
            if !folded.is_empty() {
                let folded_ms: f64 = folded.iter().map(|(_, ms)| ms).sum();
                renderer_rows.push(
                    row(
                        format!("{} more under 1 µs", folded.len()),
                        per_frame(folded_ms),
                    )
                    .nested_2(),
                );
            }
        }
        // The glue between the scopes.
        if let Some(in_scopes) = in_scopes {
            renderer_rows
                .push(row("outside the scopes", per_frame((ms - in_scopes).max(0.0))).nested());
        }
    }
    panel.set(
        Group::Renderer,
        figures.frame_work.map_or("—".to_string(), per_frame),
        renderer_rows,
        false,
    );

    let late_bad = figures.late_windows >= LATE_WINDOWS;
    let mut physics_rows = vec![
        row(
            "steps",
            format!("{:.0} steps/s", figures.steps.unwrap_or(0.0)),
        ),
        row(
            "late frames",
            format!(
                "{:.0} / s · {} total",
                figures.late_rate, figures.late_total
            ),
        )
        .bad(late_bad),
    ];
    for (name, ms) in STEP_PHASES.iter().zip(figures.step_phases) {
        if let Some(ms) = ms {
            physics_rows.push(row(*name, per_step(ms)));
        }
    }
    if let (Some(share), Some((_, workers))) = (figures.pool_share, pool) {
        let (pool_value, own_value) = match share {
            Some(pct) => (
                format!("{pct}% of task time"),
                format!("{}% of task time", 100 - pct),
            ),
            None => ("idle".to_string(), "idle".to_string()),
        };
        physics_rows.push(row("solver tasks: physics worker", own_value));
        physics_rows.push(row(
            format!("solver tasks: {workers} pool threads"),
            pool_value,
        ));
    }
    if let Some(balls) = balls {
        physics_rows.push(row("balls dropped", balls.to_string()));
    }
    panel.set(
        Group::Physics,
        figures.step_work.map_or("—".to_string(), per_step),
        physics_rows,
        late_bad,
    );

    let page_rows = match (figures.page_frame, figures.page_cues) {
        (Some(frame), Some(cues)) => vec![
            row("camera frame loop", per_frame(frame)),
            row("sound cues", per_frame(cues)),
        ],
        _ => Vec::new(),
    };
    panel.set(
        Group::Page,
        figures.page.map_or("—".to_string(), per_frame),
        page_rows,
        false,
    );
}

/// The `frame budget` group: one lane per thread plus the GPU, each its time
/// per drawn frame, against the refresh period. CPU and GPU work run in
/// parallel, so each must fit the budget on its own; they do not add up,
/// except when the CPU waits on the GPU, which shows as missed frames.
fn show_budget(
    panel: &Panel,
    figures: &Figures,
    period_ms: Option<f64>,
    pool: Option<(usize, u32)>,
) {
    let Some(budget) = period_ms else {
        panel.set(Group::Budget, "measuring…", Vec::new(), false);
        return;
    };
    // The pool threads get no lane: they run in parallel with the physics
    // worker, inside the step it times (it waits for them there).
    let lanes: Vec<(String, f64)> = [
        (Group::Gpu.label().to_string(), figures.gpu),
        (Group::Renderer.label().to_string(), figures.frame_work),
        (physics_label(pool), figures.physics_per_frame),
        (Group::Page.label().to_string(), figures.page),
    ]
    .into_iter()
    .filter_map(|(label, ms)| Some((label, ms?)))
    .collect();

    let mut headline = format!("{budget:.1} ms ({:.0} Hz)", 1000.0 / budget);
    let mut over = false;
    if let Some(largest) = lanes.iter().map(|(_, ms)| *ms).reduce(f64::max) {
        let headroom = budget - largest;
        over = headroom < 0.0;
        if over {
            headline += &format!(" · over by {:.1} ms", -headroom);
        } else {
            headline += &format!(" · headroom {headroom:.1} ms");
        }
    }
    let mut rows: Vec<Row> = lanes
        .iter()
        .map(|(label, ms)| row(label.as_str(), budget_share(*ms, budget)))
        .collect();
    if let (Some(renderer), Some(gpu)) = (figures.frame_work, figures.gpu) {
        rows.push(row(
            "latency (renderer + GPU), reference only",
            format!("{:.1} ms", renderer + gpu),
        ));
    }
    panel.set(Group::Budget, headline, rows, over);
}
