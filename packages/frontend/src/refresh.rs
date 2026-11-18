//! Display refresh rate and missed frames, for the stats panel.
//!
//! There is no web API for the screen's refresh rate, so the main thread
//! measures it: a `requestAnimationFrame` loop of its own that does nothing
//! but record the time between callbacks. The main thread is light (DOM,
//! input, audio), so it makes nearly every refresh, and the typical interval
//! is the refresh period `P` of the browser's frame clock for this window.
//! That is what the panel's `display` group shows. (It is the clock the browser
//! paces frames with; on an odd multi-monitor setup that can differ from the
//! panel's physical refresh, and no page can see past it.)
//!
//! Missed frames are counted against `P`: the renderer reports how many
//! frames it presented and the time those frames spanned (the sum of the
//! intervals between their rAF timestamps). A span of `S` ms holds `S / P`
//! refreshes, so `round(S / P) - frames` refreshes went by with no new frame
//! on screen: each one is a frame the display showed twice.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

/// Intervals kept for the estimate (about two seconds at 60 Hz).
const RING: usize = 120;
/// Intervals needed before there is an estimate at all.
const MIN_SAMPLES: usize = 30;
/// Intervals within this share of the median count as single refreshes; the
/// rest are frames the main thread itself missed (or a hidden tab) and are
/// ignored.
const INLIER_SHARE: f64 = 0.15;
/// A gap longer than this is a pause (hidden tab, debugger), not a frame.
const MAX_INTERVAL_MS: f64 = 250.0;
/// A stats window whose presented-frame span exceeds its wall-clock length by
/// more than this straddles a pause (the tab was hidden, so no frames ran):
/// it is left out of the missed count, frames and span together.
const MAX_SPAN_EXCESS_MS: f64 = 250.0;

/// The refresh-period estimate, fed by [`install`]'s rAF loop.
#[derive(Default)]
pub struct Refresh {
    intervals: VecDeque<f64>,
    last_ms: Option<f64>,
}

impl Refresh {
    fn push(&mut self, now_ms: f64) {
        if let Some(last) = self.last_ms.replace(now_ms) {
            let d = now_ms - last;
            if d > 0.0 && d < MAX_INTERVAL_MS {
                if self.intervals.len() == RING {
                    self.intervals.pop_front();
                }
                self.intervals.push_back(d);
            }
        }
    }

    /// The refresh period in ms: the mean of the intervals within
    /// [`INLIER_SHARE`] of the median. `None` until enough have been seen.
    pub fn period_ms(&self) -> Option<f64> {
        if self.intervals.len() < MIN_SAMPLES {
            return None;
        }
        let mut sorted: Vec<f64> = self.intervals.iter().copied().collect();
        sorted.sort_by(f64::total_cmp);
        let median = sorted[sorted.len() / 2];
        let inliers: Vec<f64> = sorted
            .into_iter()
            .filter(|d| (d - median).abs() <= median * INLIER_SHARE)
            .collect();
        (!inliers.is_empty()).then(|| inliers.iter().sum::<f64>() / inliers.len() as f64)
    }

    /// The panel's `refresh rate` value.
    pub fn line(&self) -> String {
        match self.period_ms() {
            Some(p) => format!("{:.0} Hz", 1000.0 / p),
            None => "measuring…".to_string(),
        }
    }
}

/// The rAF callback, held in a cell so it can schedule itself again.
type FrameLoop = Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>>;

/// Start the main thread's measuring rAF loop.
pub fn install(window: &web_sys::Window) -> Result<Rc<RefCell<Refresh>>, JsValue> {
    let refresh = Rc::new(RefCell::new(Refresh::default()));
    let cb: FrameLoop = Rc::new(RefCell::new(None));
    let next = cb.clone();
    let state = refresh.clone();
    let win = window.clone();
    *cb.borrow_mut() = Some(Closure::new(move |t: f64| {
        state.borrow_mut().push(t);
        if let Some(f) = next.borrow().as_ref() {
            let _ = win.request_animation_frame(f.as_ref().unchecked_ref());
        }
    }));
    if let Some(f) = cb.borrow().as_ref() {
        window.request_animation_frame(f.as_ref().unchecked_ref())?;
    }
    // The loop runs for the session; the closure keeps itself alive.
    std::mem::forget(cb);
    Ok(refresh)
}

/// Missed-frame bookkeeping across stats windows.
#[derive(Default)]
pub struct Missed {
    /// Signed running sum of each window's `round(S / P) - frames`. The
    /// counters are read at arbitrary moments, so one window can come out a
    /// frame high and the next a frame low; summing the signed values cancels
    /// that, and only the total is clamped.
    sum: i64,
    /// The latest window's missed frames per second (`None` before the first).
    rate: Option<f64>,
}

impl Missed {
    /// Account one stats window: `frames` presented over a span of `span_ms`
    /// (the sum of their frame intervals), `wall_ms` of wall-clock time.
    pub fn window(&mut self, frames: u32, span_ms: f64, wall_ms: f64, period_ms: Option<f64>) {
        let Some(p) = period_ms else { return };
        if span_ms > wall_ms + MAX_SPAN_EXCESS_MS || wall_ms <= 0.0 {
            return;
        }
        let missed = (span_ms / p).round() as i64 - frames as i64;
        self.sum += missed;
        self.rate = Some(missed.max(0) as f64 / (wall_ms / 1000.0));
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// The panel's `missed frames` value.
    pub fn line(&self) -> String {
        match self.rate {
            Some(rate) => format!("{rate:.0} / s · {} total", self.sum.max(0)),
            None => "—".to_string(),
        }
    }
}
