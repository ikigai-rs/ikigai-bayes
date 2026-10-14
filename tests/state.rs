//! The state directory: a model survives the host, a missing or corrupt model is refused by
//! name, only a new model starts empty, and one writer holds a directory.

mod common;

use common::*;
use ikigai_bayes::{Bayes, ModelConfig};
use ikigai_core::Verb;

fn open(dir: &std::path::Path) -> Result<Bayes, ikigai_bayes::OpenError> {
    Bayes::open(dir, [ModelConfig::new("contact")])
}

#[test]
fn a_new_model_is_created_and_registered_at_open() {
    let s = Scratch::new();
    let bayes = open(s.path()).unwrap();
    drop(bayes);
    assert!(s.path().join("contact.json").is_file());
    assert_eq!(
        std::fs::read_to_string(s.path().join("models")).unwrap(),
        "contact\n"
    );
    // Nothing left behind by the atomic writes.
    assert!(!s.path().join("contact.json.tmp").exists());
}

#[test]
fn a_model_survives_the_host() {
    let s = Scratch::new();
    let (stats, scored) = {
        let bayes = open(s.path()).unwrap();
        let k = kernel(&bayes);
        train(&k, "contact", "cheap pills buy now", "spam");
        train(&k, "contact", "lunch on tuesday", "ham");
        (
            text(&k, Verb::Source, "urn:bayes:contact:stats", &[]),
            score(&k, "contact", "cheap lunch pills"),
        )
    };
    let bayes = open(s.path()).unwrap();
    let k = kernel(&bayes);
    assert_eq!(
        text(&k, Verb::Source, "urn:bayes:contact:stats", &[]),
        stats
    );
    assert_eq!(score(&k, "contact", "cheap lunch pills"), scored);
}

#[test]
fn a_registered_model_whose_file_is_gone_is_refused_by_name() {
    let s = Scratch::new();
    drop(open(s.path()).unwrap());
    std::fs::remove_file(s.path().join("contact.json")).unwrap();
    let err = open(s.path()).unwrap_err().to_string();
    assert!(err.contains("model `contact`"), "{err}");
    assert!(err.contains("is missing"), "{err}");
    assert!(err.contains("refusing to start it empty"), "{err}");
}

#[test]
fn a_corrupt_model_is_refused_by_name() {
    let s = Scratch::new();
    drop(open(s.path()).unwrap());
    std::fs::write(s.path().join("contact.json"), b"{\"format\":").unwrap();
    let err = open(s.path()).unwrap_err().to_string();
    assert!(
        err.contains("model `contact`: its file is refused"),
        "{err}"
    );
}

#[test]
fn a_file_without_a_registry_entry_is_read_and_registered() {
    let s = Scratch::new();
    {
        let bayes = open(s.path()).unwrap();
        train(&kernel(&bayes), "contact", "cheap pills", "spam");
    }
    std::fs::remove_file(s.path().join("models")).unwrap();
    let bayes = open(s.path()).unwrap();
    assert!(text(
        &kernel(&bayes),
        Verb::Source,
        "urn:bayes:contact:stats",
        &[]
    )
    .contains("spam 1 ham 0"));
    assert_eq!(
        std::fs::read_to_string(s.path().join("models")).unwrap(),
        "contact\n"
    );
}

#[test]
fn one_writer_per_directory() {
    let s = Scratch::new();
    let held = open(s.path()).unwrap();
    let err = open(s.path()).unwrap_err().to_string();
    assert!(err.contains("held by another writer"), "{err}");
    assert!(
        err.contains(&format!("pid {}", std::process::id())),
        "{err}"
    );
    drop(held);
    assert!(open(s.path()).is_ok(), "the lock goes with its holder");
}

#[test]
fn a_failed_write_changes_nothing() {
    let s = Scratch::new();
    let bayes = open(s.path()).unwrap();
    let k = kernel(&bayes);
    train(&k, "contact", "lunch on tuesday", "ham");
    let before = text(&k, Verb::Source, "urn:bayes:contact:stats", &[]);
    // A directory where the temp file must go makes the atomic write fail.
    std::fs::create_dir(s.path().join("contact.json.tmp")).unwrap();
    let err = issue(
        &k,
        Verb::Sink,
        "urn:bayes:contact:train",
        &[("content", "cheap pills"), ("label", "spam")],
    )
    .unwrap_err();
    assert!(matches!(err, ikigai_core::Error::Endpoint(_)), "{err:?}");
    std::fs::remove_dir(s.path().join("contact.json.tmp")).unwrap();
    assert_eq!(
        text(&k, Verb::Source, "urn:bayes:contact:stats", &[]),
        before
    );
    drop(k);
    drop(bayes);
    let reopened = open(s.path()).unwrap();
    assert_eq!(
        text(
            &kernel(&reopened),
            Verb::Source,
            "urn:bayes:contact:stats",
            &[]
        ),
        before
    );
}
