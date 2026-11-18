//! How the panel shows a value: the numbers apart from the units and words
//! around them, and a frame-budget lane's bar.
//!
//! No dependencies, so its tests run on the host, which the crate itself
//! does not build for. From the repo root: `rustc --edition 2021 --test
//! packages/frontend/src/stats/value.rs -o target/stats-value-tests &&
//! target/stats-value-tests`.

/// A lane's share of the frame budget from which its bar warns.
const SHARE_WARN: f64 = 0.8;

/// A piece of a value: a number (shown bright) or the text around it.
#[derive(Debug, PartialEq, Eq)]
pub enum Piece<'a> {
    Num(&'a str),
    Unit(&'a str),
}

/// Splits `text` into numbers and the rest. A number is an optional `<`,
/// a digit, then digits and dots, then an optional `%` (`7.4`, `<0.1`,
/// `<1%`). Empty pieces are left out.
pub fn pieces(text: &str) -> Vec<Piece<'_>> {
    let bytes = text.as_bytes();
    let is_num = |b: &u8| b.is_ascii_digit() || *b == b'.';
    let mut pieces = Vec::new();
    let mut unit_start = 0;
    let mut i = 0;
    while i < bytes.len() {
        let digit = i + usize::from(bytes[i] == b'<');
        if !bytes.get(digit).is_some_and(u8::is_ascii_digit) {
            i += 1;
            continue;
        }
        let mut end = digit + 1;
        while bytes.get(end).is_some_and(is_num) {
            end += 1;
        }
        if bytes.get(end) == Some(&b'%') {
            end += 1;
        }
        // Only ever split before `<` or a digit and after a digit, `.` or
        // `%`: all ASCII, so always on a char boundary.
        if unit_start < i {
            pieces.push(Piece::Unit(&text[unit_start..i]));
        }
        pieces.push(Piece::Num(&text[i..end]));
        unit_start = end;
        i = end;
    }
    if unit_start < bytes.len() {
        pieces.push(Piece::Unit(&text[unit_start..]));
    }
    pieces
}

/// A lane's bar class for its `share` of the budget: red over it, amber from
/// `SHARE_WARN`, neutral below.
pub fn bar_state(share: f64) -> Option<&'static str> {
    if share > 1.0 {
        Some("stats-over")
    } else if share >= SHARE_WARN {
        Some("stats-warn")
    } else {
        None
    }
}

/// The bar's fill, as a CSS width: the share, capped at the whole bar.
pub fn bar_width(share: f64) -> String {
    format!("{}%", share.min(1.0) * 100.0)
}

#[cfg(test)]
mod tests {
    use super::Piece::{Num, Unit};
    use super::*;

    #[test]
    fn splits_the_panels_values() {
        assert_eq!(
            pieces("16.7 ms (60 Hz) · headroom 9.6 ms"),
            [
                Num("16.7"),
                Unit(" ms ("),
                Num("60"),
                Unit(" Hz) · headroom "),
                Num("9.6"),
                Unit(" ms"),
            ]
        );
        assert_eq!(
            pieces("<0.1 / 16.7 ms (<1%)"),
            [
                Num("<0.1"),
                Unit(" / "),
                Num("16.7"),
                Unit(" ms ("),
                Num("<1%"),
                Unit(")"),
            ]
        );
        assert_eq!(
            pieces("0 / s · 0 total"),
            [Num("0"), Unit(" / s · "), Num("0"), Unit(" total")]
        );
        assert_eq!(pieces("7.4 µs / frame"), [Num("7.4"), Unit(" µs / frame")]);
        assert_eq!(
            pieces("43% of task time"),
            [Num("43%"), Unit(" of task time")]
        );
        assert_eq!(pieces("n/a"), [Unit("n/a")]);
        assert_eq!(pieces("—"), [Unit("—")]);
        assert_eq!(pieces("measuring…"), [Unit("measuring…")]);
        assert!(pieces("").is_empty());
    }

    #[test]
    fn a_lone_less_than_is_text() {
        assert_eq!(pieces("a < b"), [Unit("a < b")]);
        assert_eq!(pieces("<<5"), [Unit("<"), Num("<5")]);
    }

    #[test]
    fn bar_thresholds() {
        assert_eq!(bar_state(0.0), None);
        assert_eq!(bar_state(0.79), None);
        assert_eq!(bar_state(0.8), Some("stats-warn"));
        assert_eq!(bar_state(1.0), Some("stats-warn"));
        assert_eq!(bar_state(1.01), Some("stats-over"));
        assert_eq!(bar_width(0.5), "50%");
        assert_eq!(bar_width(1.0), "100%");
        assert_eq!(bar_width(2.5), "100%");
    }
}
