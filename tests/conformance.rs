//! The module recipe as one test: `ikigai-conformance` walks every resource this crate binds and
//! reports every violation at once.
//!
//! # What the suite cannot say, and this file says instead
//!
//! - **`space(&bayes)` is HOST-named**: its doors are the models the host opened.
//! - **Score and stats are `cacheable`** on the model's two threads (`…:train`, `…:untrain`),
//!   which the kernel cuts after every write through them. `tests/resources.rs` measures the
//!   cut with `Kernel::is_cached`.
//! - **`urn:bayes:models` is `pure`** for the life of the host: the models are fixed at open.
//! - **The write doors are held to REQUIRED** (`required_on_mutating`): `train` without its
//!   `content` or its `label` must be refused, typed. The state is a scratch store, so a drop
//!   wrongly accepted would teach only the scratch model.
//! - **`untrain` declares no required input**: it takes `content` OR `digest=`, which an
//!   `ArgSpec` cannot spell as "exactly one of". Its refusals are pinned in
//!   `tests/resources.rs` (`bad_calls_are_refused_typed`).

mod common;

use std::sync::Arc;

use common::*;
use ikigai_conformance::Suite;
use ikigai_core::Kernel;

const SPACE_LABEL: &str = "ikigai_bayes::space(&bayes)";
const MODELS: [&str; 2] = ["contact", "other"];

fn walk() -> ikigai_conformance::Report {
    let bayes = two_models();
    let space = Arc::new(ikigai_bayes::space(&bayes));
    let kernel = Kernel::with_meta_renderer(space.clone(), Arc::new(ikigai_vocab::TurtleRenderer));
    // Taught before the walk, so the pipeline probe's untrain of the generic sample `x` finds a
    // message to remove whichever order the walk fires the doors in, and the score's Turtle face
    // has clues to skolemize.
    for model in MODELS {
        train(&kernel, model, "x", "spam");
    }
    let mut suite = Suite::new()
        .host_named_space(SPACE_LABEL, space)
        .required_on_mutating()
        .pure("bayes-models");
    for model in MODELS {
        suite = suite
            .cacheable(format!("bayes-{model}-score"))
            .cacheable(format!("bayes-{model}-stats"));
    }
    suite.run_blocking(&kernel)
}

#[test]
fn conforms() {
    let report = walk();
    println!("{report}");
    report.assert_clean();
    assert!(
        report
            .to_string()
            .contains(&format!("space: {SPACE_LABEL} host-named\n")),
        "{report}"
    );
}

/// A clean report over a walk that reached nothing would look exactly like a clean report over a
/// walk that reached everything. This pins the list.
#[test]
fn the_walk_reaches_every_resource_this_crate_binds() {
    let report = walk();
    let mut walked: Vec<&str> = report.walked.iter().map(String::as_str).collect();
    walked.sort_unstable();
    let mut expected = vec!["bayes-models".to_string()];
    for model in MODELS {
        for part in ["score", "train", "untrain", "stats"] {
            expected.push(format!("bayes-{model}-{part}"));
        }
    }
    expected.sort_unstable();
    assert_eq!(walked, expected);
}
