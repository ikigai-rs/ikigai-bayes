//! Helpers shared by the integration tests.
#![allow(dead_code)] // each test binary uses a different subset

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use futures::executor::block_on;
use ikigai_bayes::{Bayes, ModelConfig};
use ikigai_core::{ArgRef, Capability, Iri, Kernel, Representation, Request, Result, Verb};

/// A kernel over `bayes`'s space, with a JSON Meta face (so engine-facing behavior is the
/// behavior a host sees).
pub fn kernel(bayes: &Bayes) -> Kernel {
    Kernel::with_meta_renderer(
        Arc::new(ikigai_bayes::space(bayes)),
        Arc::new(ikigai_vocab::TurtleRenderer),
    )
}

/// Two models in memory: `contact`, and `other` to show what does NOT cross between models.
pub fn two_models() -> Bayes {
    Bayes::in_memory([ModelConfig::new("contact"), ModelConfig::new("other")]).unwrap()
}

pub fn request(verb: Verb, iri: &str, args: &[(&str, &str)]) -> Request {
    let mut request = Request::new(verb, Iri::parse(iri).unwrap());
    for (name, value) in args {
        request = request.with_arg(*name, ArgRef::Inline(value.as_bytes().to_vec()));
    }
    request
}

pub fn issue_as(
    kernel: &Kernel,
    cap: &Capability,
    verb: Verb,
    iri: &str,
    args: &[(&str, &str)],
) -> Result<Representation> {
    block_on(kernel.issue(request(verb, iri, args), cap))
}

pub fn issue(
    kernel: &Kernel,
    verb: Verb,
    iri: &str,
    args: &[(&str, &str)],
) -> Result<Representation> {
    issue_as(kernel, &Capability::root(), verb, iri, args)
}

pub fn text(kernel: &Kernel, verb: Verb, iri: &str, args: &[(&str, &str)]) -> String {
    let rep = issue(kernel, verb, iri, args).unwrap_or_else(|e| panic!("{verb:?} {iri}: {e}"));
    String::from_utf8(rep.bytes).unwrap()
}

pub fn train(kernel: &Kernel, model: &str, content: &str, label: &str) -> String {
    text(
        kernel,
        Verb::Sink,
        &format!("urn:bayes:{model}:train"),
        &[("content", content), ("label", label)],
    )
}

pub fn score(kernel: &Kernel, model: &str, content: &str) -> String {
    text(
        kernel,
        Verb::Source,
        &format!("urn:bayes:{model}:score"),
        &[("content", content)],
    )
}

/// A fresh scratch directory under the system temp dir, removed on drop.
pub struct Scratch(PathBuf);

impl Scratch {
    pub fn new() -> Scratch {
        static N: AtomicUsize = AtomicUsize::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "ikigai-bayes-{}-{}-{nanos}",
            std::process::id(),
            N.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
