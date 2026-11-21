//! The About button (bottom centre) and its modal: the title and the link to
//! the template on GitHub, then two tabs: About (what the demo is) and Stats
//! (what the Stats panel's numbers mean).

use dominator::{clone, html, with_node, Dom};
use futures_signals::signal::{Mutable, SignalExt};
use wasm_bindgen::JsCast;
use web_sys::HtmlElement;

use crate::main_thread::{keep_keys_in_modal, QUALITY_NOTICE_ID};
use crate::stats::Lane;

const REPO: &str = "https://github.com/awsm-fun/awsm-template-physics-multithreaded";

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub enum Tab {
    #[default]
    About,
    Stats,
}

impl Tab {
    const ALL: [Tab; 2] = [Tab::About, Tab::Stats];

    fn label(self) -> &'static str {
        match self {
            Tab::About => "About",
            Tab::Stats => "Stats",
        }
    }

    fn tab_id(self) -> String {
        format!("about-tab-{}", self.label().to_lowercase())
    }

    fn panel_id(self) -> String {
        format!("about-panel-{}", self.label().to_lowercase())
    }

    fn focus(self) {
        let tab = web_sys::window()
            .and_then(|w| w.document())
            .and_then(|d| d.get_element_by_id(&self.tab_id()))
            .and_then(|el| el.dyn_into::<HtmlElement>().ok());
        if let Some(tab) = tab {
            let _ = tab.focus();
        }
    }
}

/// Whether the modal is open, and on which tab.
#[derive(Clone, Default)]
pub struct About {
    open: Mutable<bool>,
    tab: Mutable<Tab>,
}

impl About {
    /// Opens the modal on `tab`; the tab takes the keyboard focus.
    pub fn show(&self, tab: Tab) {
        self.tab.set_neq(tab);
        self.open.set_neq(true);
    }

    /// The bottom-centre chip. Outside the canvas, so a click on it never
    /// drops a ball.
    pub fn button(&self) -> Dom {
        let about = self.clone();
        html!("button", {
            .attr("type", "button")
            .class("about-btn")
            .text("About")
            .event(move |_: dominator::events::Click| about.show(Tab::About))
        })
    }

    pub fn modal(&self) -> Dom {
        let Self { open, tab } = self;
        html!("div", {
            .class("about-overlay")
            .attr("hidden", "")
            .event_with_options(&dominator::EventOptions::bubbles(), keep_keys_in_modal)
            // Only a click on the backdrop itself closes it, not one in the card.
            .event(clone!(open => move |e: dominator::events::Click| {
                let on_backdrop = e
                    .dyn_target::<web_sys::Element>()
                    .is_some_and(|el| el.class_list().contains("about-overlay"));
                if on_backdrop {
                    open.set_neq(false);
                }
            }))
            .global_event(clone!(open => move |e: dominator::events::KeyDown| {
                if e.key() == "Escape" && !quality_notice_shown() {
                    open.set_neq(false);
                }
            }))
            // Shown and focused in one step: a hidden tab cannot take focus.
            .with_node!(overlay => {
                .future(open.signal().for_each(clone!(tab => move |open| {
                    overlay.set_hidden(!open);
                    if open {
                        tab.get().focus();
                    }
                    async {}
                })))
            })
            .child(html!("div", {
                .class("about-modal")
                .attr("role", "dialog")
                .attr("aria-modal", "true")
                .attr("aria-labelledby", "about-title")
                .child(html!("button", {
                    .attr("type", "button")
                    .class("about-close")
                    .attr("aria-label", "Close")
                    .text("×")
                    .event(clone!(open => move |_: dominator::events::Click| open.set_neq(false)))
                }))
                .child(html!("header", {
                    .class("about-header")
                    .child(html!("h2", {
                        .attr("id", "about-title")
                        .text("Multithreaded Physics Demo")
                    }))
                    .child(link(REPO, "View the template on GitHub ↗").class("about-github").into_dom())
                }))
                .child(html!("div", {
                    .class("about-tabs")
                    .attr("role", "tablist")
                    .attr("aria-label", "About this demo")
                    .children(Tab::ALL.map(|t| html!("button", {
                        .attr("type", "button")
                        .class("about-tab")
                        .attr("role", "tab")
                        .attr("id", &t.tab_id())
                        .attr("aria-controls", &t.panel_id())
                        .attr_signal("aria-selected", tab.signal().map(move |s| if s == t { "true" } else { "false" }))
                        .attr_signal("tabindex", tab.signal().map(move |s| if s == t { "0" } else { "-1" }))
                        .text(t.label())
                        .event(clone!(tab => move |_: dominator::events::Click| tab.set_neq(t)))
                    })))
                    // Preventable, so that Home and End do not also scroll.
                    .event_with_options(&dominator::EventOptions::preventable(), clone!(tab => move |e: dominator::events::KeyDown| {
                        let n = Tab::ALL.len();
                        let current = Tab::ALL.iter().position(|&t| t == tab.get()).unwrap_or(0);
                        let next = match e.key().as_str() {
                            "ArrowRight" => (current + 1) % n,
                            "ArrowLeft" => (current + n - 1) % n,
                            "Home" => 0,
                            "End" => n - 1,
                            _ => return,
                        };
                        e.prevent_default();
                        tab.set_neq(Tab::ALL[next]);
                        Tab::ALL[next].focus();
                    }))
                }))
                .child(panel(tab, Tab::About, about_tab()))
                .child(panel(tab, Tab::Stats, stats_tab()))
            }))
        })
    }
}

/// The quality notice opens over this modal; while it is up, Escape is its.
fn quality_notice_shown() -> bool {
    web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.get_element_by_id(QUALITY_NOTICE_ID))
        .and_then(|el| el.dyn_into::<HtmlElement>().ok())
        .is_some_and(|el| el.offset_width() > 0)
}

fn panel(selected: &Mutable<Tab>, tab: Tab, content: Vec<Dom>) -> Dom {
    html!("div", {
        .class("about-panel")
        .attr("role", "tabpanel")
        .attr("id", &tab.panel_id())
        .attr("aria-labelledby", &tab.tab_id())
        .attr("tabindex", "0")
        .prop_signal("hidden", selected.signal().map(move |s| s != tab))
        .children(content)
    })
}

fn about_tab() -> Vec<Dom> {
    vec![
        html!("p", {
            .text("Roll a red ball around a table and drop more balls onto it; every \
                   bounce and knock makes a sound in 3D. It is a template to start your \
                   own game from, with the physics on threads of its own.")
        }),
        heading("Controls"),
        keys(&[
            ("W A S D or arrows", "roll the ball, relative to the camera"),
            ("Space", "jump"),
            ("Click the table", "drop a ball there"),
            ("Right-drag", "orbit the camera"),
            ("Wheel", "zoom"),
            ("Swipe (touch)", "fling the ball"),
            ("Tap (touch)", "drop a ball"),
            (
                "Stats, Settings",
                "what each thread is doing; resolution and anti-aliasing",
            ),
        ]),
        heading("How it works"),
        html!("p", {
            .text("The threads share one block of memory. This page owns the canvas, \
                   the input and the sound (")
            .child(link("https://audio.awsm.fun", "AwsmAudio").into_dom())
            .text("). A physics worker runs ")
            .child(link("https://github.com/erincatto/box3d", "Box3D").into_dom())
            .text(" at a fixed 240 steps a second, with a pool of workers sharing its \
                   solver's work. A Rust WebGPU renderer (")
            .child(link("https://scene.awsm.fun", "AwsmRenderer").into_dom())
            .text(") in another worker draws it, blending between steps so motion stays \
                   smooth at any refresh rate. Nothing that happens every frame goes \
                   through postMessage.")
        }),
        heading("Make it yours"),
        prose(
            "p",
            "Start with `packages/frontend/src/physics_thread.rs`: the game (forces, ball \
             drops, sound cues). For your own scene and sounds, see \"Swapping in your own \
             scene\" in the README.",
        ),
        html!("p", {
            .class("about-more")
            .text("Built with ❤ by David Komer. The README has the rest. Part of the ")
            .child(link("https://awsm.fun", "Awsm").into_dom())
            .text(" project.")
        }),
    ]
}

fn stats_tab() -> Vec<Dom> {
    vec![
        html!("p", {
            .text("The Stats button (bottom left) opens a panel: a frame budget first, \
                   then one group per thread. Each group's header shows its main number; \
                   click a group to open its breakdown. A breakdown is only measured while \
                   its group is open, so the panel costs nothing when it is closed.")
        }),
        units(
            "`ms / frame` is time per rendered frame. `µs / step` is time per simulation \
             step (1000 µs = 1 ms). `/ s` is per second.",
        ),
        heading("frame budget"),
        terms(&[
            (
                "budget",
                "The time one display refresh allows: 16.7 ms at 60 Hz, 8.3 ms at 120 Hz. \
                 It comes from your display's measured refresh rate.",
            ),
            (
                "headroom",
                "The budget minus the largest lane below. The GPU and the threads work in \
                 parallel, so their times do not add up: each one has to fit the budget on \
                 its own. If it reads `over by …`, that lane cannot keep up with the display.",
            ),
            (
                "the lanes",
                "One row for the GPU and one per thread: the time it uses per frame, out of \
                 the budget. Here: the GPU, the renderer, physics and the page. The physics \
                 worker's lane is its whole step, the pool's work included, times the steps \
                 in a frame.",
            ),
            (
                "latency (renderer + GPU), reference only",
                "Roughly how long one frame takes from the renderer starting it to the GPU \
                 finishing it. Not what the budget is checked against.",
            ),
        ]),
        html!("p", {
            .text("Missed frames while there is headroom mean something is waiting rather \
                   than working, for example the CPU waiting on the GPU.")
        }),
        heading("display"),
        terms(&[
            (
                "refresh rate",
                "How often your display refreshes, as the browser paces this page.",
            ),
            (
                "frames presented",
                "New frames shown per second (fps). Never more than the refresh rate.",
            ),
            (
                "missed frames",
                "Refreshes that went by without a new frame. A steady count is visible \
                 stutter: lower the resolution or the anti-aliasing in Settings.",
            ),
        ]),
        lane_heading("GPU", Lane::Gpu),
        terms(&[
            (
                "GPU time per frame",
                "How long the GPU works on a frame, from timestamp queries. Reads `n/a` \
                 where the browser or the GPU does not offer them.",
            ),
            (
                "the rows",
                "One per render pass. Some GPUs (Apple's) run passes overlapped, so the rows \
                 can add up to more than the frame.",
            ),
        ]),
        lane_heading("renderer (render worker)", Lane::Renderer),
        terms(&[
            ("the header", "The render thread's CPU time per frame."),
            (
                "read app state + interpolate",
                "Reads the two latest simulation steps and blends between them for this \
                 frame's exact moment.",
            ),
            (
                "animations + particles",
                "Advances the scene's authored clips, texture flows and particle emitters \
                 by the frame's time.",
            ),
            (
                "set camera, update transforms",
                "Sets the view, and turns the new poses into world matrices.",
            ),
            (
                "render (encode + submit)",
                "The renderer's own work, split into `prepare` (uploading this frame's data \
                 to the GPU), `passes` (encoding each render pass), `submit` (handing the \
                 frame to the GPU) and `outside the scopes` (the little that is left \
                 untimed). The smallest rows are folded into \"N more under 1 µs\".",
            ),
        ]),
        lane_heading(
            "physics (Box3D, physics worker + N pool threads)",
            Lane::Sim,
        ),
        terms(&[
            (
                "the header",
                "CPU time of one fixed simulation step, including the time the physics \
                 worker waits for the pool.",
            ),
            (
                "steps",
                "Steps per second. Fixed (240), whatever the display does; the renderer \
                 blends between steps.",
            ),
            (
                "late frames",
                "Frames drawn before physics had finished the moment they show; such a \
                 frame holds the previous pose. The row and the header turn red if it keeps \
                 happening.",
            ),
            (
                "input + ball drops, Box3D step, contacts + sound cues, publish poses",
                "The parts of a step, in order.",
            ),
            (
                "solver tasks: physics worker, solver tasks: N pool threads",
                "How Box3D's solver work splits between the physics worker and the pool. \
                 The physics worker helps while it waits, so it takes most of a small \
                 scene; drop more balls and the pool's share grows.",
            ),
            ("balls dropped", "How many balls you have dropped."),
        ]),
        lane_heading("page (main thread)", Lane::Page),
        terms(&[(
            "camera frame loop, sound cues",
            "The page's own work per frame: the camera and the sound cues it plays. It \
             should stay tiny; the heavy work is in the workers.",
        )]),
    ]
}

fn link(href: &str, label: &str) -> dominator::DomBuilder<HtmlElement> {
    dominator::DomBuilder::new_html("a")
        .attr("href", href)
        .attr("target", "_blank")
        .attr("rel", "noopener")
        .text(label)
}

fn heading(text: &str) -> Dom {
    html!("h3", { .text(text) })
}

/// A Stats-tab heading for a group with a lane: its dot, as in the panel.
fn lane_heading(text: &str, lane: Lane) -> Dom {
    html!("h3", { .attr("data-lane", lane.attr()).text(text) })
}

/// `text` in a `tag` element, each `backticked` part set as code.
fn prose(tag: &str, text: &str) -> Dom {
    marked(tag, text, |part| html!("code", { .text(part) }))
}

/// A paragraph about units, each `backticked` unit a pill.
fn units(text: &str) -> Dom {
    marked(
        "p",
        text,
        |unit| html!("span", { .class("about-unit").text(unit) }),
    )
}

/// `text` in a `tag` element, each `backticked` part made by `mark`.
fn marked(tag: &str, text: &str, mark: fn(&str) -> Dom) -> Dom {
    html!(tag, {
        .children(text.split('`').enumerate().map(|(i, part)| {
            if i % 2 == 1 {
                mark(part)
            } else {
                dominator::text(part)
            }
        }))
    })
}

/// Key → what it does.
fn keys(rows: &[(&str, &str)]) -> Dom {
    html!("table", {
        .class("about-keys")
        .child(html!("tbody", {
            .children(rows.iter().map(|(key, does)| html!("tr", {
                .child(html!("th", { .attr("scope", "row").text(key) }))
                .child(html!("td", { .text(does) }))
            })))
        }))
    })
}

/// A term from the panel, then what it means.
fn terms(items: &[(&str, &str)]) -> Dom {
    html!("dl", {
        .children(items.iter().flat_map(|(term, meaning)| [
            html!("dt", { .text(term) }),
            prose("dd", meaning),
        ]))
    })
}
