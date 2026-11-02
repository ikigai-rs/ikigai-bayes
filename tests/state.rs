//! The state directory: a model survives the host, a missing or corrupt model is refused by
//! name, only a new model starts empty, and one writer holds a directory. Since Hypothesis H
//! (ledger #2384) the directory is reached through a private `urn:file:` mount, so these tests
//! are also the proof that the resources answer what the files did.

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
    // Nothing left behind by the atomic writes: no staging file beside the three names.
    let mut names: Vec<String> = std::fs::read_dir(s.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    assert_eq!(names, ["LOCK", "contact.json", "models"]);
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
    // A directory where the model's file must go makes the write fail (a rename cannot
    // replace a directory with a file). The file is set aside, and put back after.
    let file = s.path().join("contact.json");
    let aside = s.path().join("contact.aside");
    std::fs::rename(&file, &aside).unwrap();
    std::fs::create_dir(&file).unwrap();
    let err = issue(
        &k,
        Verb::Sink,
        "urn:bayes:contact:train",
        &[("content", "cheap pills"), ("label", "spam")],
    )
    .unwrap_err();
    assert!(matches!(err, ikigai_core::Error::Endpoint(_)), "{err:?}");
    std::fs::remove_dir(&file).unwrap();
    std::fs::rename(&aside, &file).unwrap();
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

/// The same models through a `urn:file:` mount the HOST binds, at a base of its choosing: what
/// `DirStore` writes is what a `ResourceStore` over that directory reads, byte for byte.
#[test]
fn a_resource_store_reads_what_the_state_directory_wrote() {
    use ikigai_bayes::store::{ResourceStore, Store};
    use ikigai_core::{Capability, Kernel};
    use std::sync::Arc;

    let s = Scratch::new();
    let dir = s.path().join("state");
    let stats = {
        let bayes = open(&dir).unwrap();
        let k = kernel(&bayes);
        train(&k, "contact", "cheap pills buy now", "spam");
        text(&k, Verb::Source, "urn:bayes:contact:stats", &[])
    };
    let on_disk = std::fs::read(dir.join("contact.json")).unwrap();

    // The host's mount is jailed one level up, so the models live under `urn:file:state/`.
    let files = Arc::new(Kernel::new(Arc::new(ikigai_fs::space(s.path()))));
    let store = ResourceStore::open(files, "urn:file:state/", Capability::root()).unwrap();
    assert_eq!(store.load("contact").unwrap(), Some(on_disk));
    assert_eq!(store.load("other").unwrap(), None, "never created: new");

    let bayes = Bayes::with_store(Arc::new(store), [ModelConfig::new("contact")]).unwrap();
    let k = kernel(&bayes);
    assert_eq!(
        text(&k, Verb::Source, "urn:bayes:contact:stats", &[]),
        stats
    );
    // A train through it lands in the same file.
    train(&k, "contact", "lunch on tuesday", "ham");
    assert!(text(&k, Verb::Source, "urn:bayes:contact:stats", &[]).contains("spam 1 ham 1"));
    drop(k);
    drop(bayes);
    let reopened = open(&dir).unwrap();
    assert!(text(
        &kernel(&reopened),
        Verb::Source,
        "urn:bayes:contact:stats",
        &[]
    )
    .contains("spam 1 ham 1"));
}

/// The store writes only under the capability it was handed: a read-only grant reads the
/// models, and a train is refused rather than written around the grant.
#[test]
fn a_resource_store_writes_only_what_its_capability_grants() {
    use ikigai_bayes::store::{ResourceStore, Store};
    use ikigai_core::{Capability, Kernel};
    use std::sync::Arc;

    let s = Scratch::new();
    drop(open(s.path()).unwrap());
    let root = s.path().to_str().expect("a UTF-8 scratch path");
    let files = Arc::new(Kernel::new(Arc::new(ikigai_fs::space(s.path()))));
    let read_only = Capability::scoped([format!("urn:cap:fs:read:{root}")]);
    let store = ResourceStore::open(files, "urn:file:", read_only).unwrap();
    assert!(store.load("contact").unwrap().is_some());
    let err = store.save("contact", b"{}").unwrap_err().to_string();
    assert!(err.contains("write urn:file:contact.json"), "{err}");
}
