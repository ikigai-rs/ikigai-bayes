mod common;
use common::*;
use ikigai_bayes::{Bayes, ModelConfig};
use ikigai_core::{Capability, Verb};
use serde_json::Value;

fn json(k: &ikigai_core::Kernel, model: &str, part: &str, args: &[(&str, &str)]) -> Value {
    let mut args = args.to_vec();
    args.push(("as", "application/json"));
    serde_json::from_str(&text(
        k,
        Verb::Source,
        &format!("urn:bayes:{model}:{part}"),
        &args,
    ))
    .unwrap()
}

#[test]
fn stats_and_score_identify_the_trained_state() {
    let bayes = two_models();
    let k = kernel(&bayes);
    let initial = json(&k, "contact", "stats", &[])["modelDigest"].clone();
    assert!(
        initial.as_str().is_some(),
        "stats has no model-state digest"
    );
    assert_eq!(initial, json(&k, "other", "stats", &[])["modelDigest"]);
    train(&k, "contact", "announcement", "spam");
    let trained = json(&k, "contact", "stats", &[])["modelDigest"].clone();
    assert_ne!(trained, initial);
    let probe = request(
        Verb::Source,
        "urn:bayes:contact:score",
        &[("content", "announcement"), ("as", "application/json")],
    );
    assert_eq!(
        json(&k, "contact", "score", &[("content", "announcement")])["modelDigest"],
        trained
    );
    assert!(k.is_cached(&probe, &Capability::root()));
    train(&k, "contact", "announcement", "spam");
    assert_eq!(trained, json(&k, "contact", "stats", &[])["modelDigest"]);
    train(&k, "other", "announcement", "spam");
    assert_eq!(trained, json(&k, "other", "stats", &[])["modelDigest"]);
    train(&k, "contact", "announcement", "ham");
    assert!(!k.is_cached(&probe, &Capability::root()));
    let moved = json(&k, "contact", "stats", &[])["modelDigest"].clone();
    assert_ne!(moved, trained);
    assert_eq!(
        json(&k, "contact", "score", &[("content", "announcement")])["modelDigest"],
        moved
    );
    text(
        &k,
        Verb::Delete,
        "urn:bayes:contact:untrain",
        &[("content", "announcement")],
    );
    assert!(!k.is_cached(&probe, &Capability::root()));
    assert_eq!(initial, json(&k, "contact", "stats", &[])["modelDigest"]);
    assert_eq!(
        initial,
        json(&k, "contact", "score", &[("content", "announcement")])["modelDigest"]
    );
}

#[test]
fn yes_no_labels_reach_every_face_and_the_contract() {
    let bayes =
        Bayes::in_memory([ModelConfig::new("question").labels("yes", "no").unwrap()]).unwrap();
    let k = kernel(&bayes);
    let meta = text(
        &k,
        Verb::Meta,
        "urn:bayes:question:train",
        &[("as", "application/json")],
    );
    assert!(meta.contains("yes") && meta.contains("no"), "{meta}");
    assert!(
        !meta.contains("\"spam\"") && !meta.contains("\"ham\""),
        "{meta}"
    );
    assert!(train(&k, "question", "cheap pills buy now", "yes").contains("as yes"));
    assert!(train(&k, "question", "lunch on tuesday", "no").contains("messages yes 1 no 1"));
    let err = issue(
        &k,
        Verb::Sink,
        "urn:bayes:question:train",
        &[("content", "x"), ("label", "spam")],
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("yes") && err.contains("no"), "{err}");
    let scored = json(&k, "question", "score", &[("content", "cheap pills")]);
    assert_eq!(scored["class"], "yes");
    assert_eq!(
        scored["cutoffs"],
        serde_json::json!({"no": 0.2, "yes": 0.9})
    );
    assert!(!scored["clues"].as_array().unwrap().is_empty());
    assert!(score(&k, "question", "cheap pills").starts_with("yes 0.920316\n"));
    for face in ["text/plain", "text/turtle"] {
        let scored_text = text(
            &k,
            Verb::Source,
            "urn:bayes:question:score",
            &[("content", "cheap pills"), ("as", face)],
        );
        assert!(scored_text.contains(scored["modelDigest"].as_str().unwrap()));
        assert!(!scored_text.contains("spam") && !scored_text.contains("ham"));
    }
    assert_eq!(
        json(&k, "question", "score", &[("content", "lunch tuesday")])["class"],
        "no"
    );
    assert_eq!(
        json(&k, "question", "score", &[("content", "unseen")])["class"],
        "unsure"
    );
    let stats = json(&k, "question", "stats", &[]);
    assert_eq!(stats["messages"], serde_json::json!({"no": 1, "yes": 1}));
    assert_eq!(stats["modelDigest"], scored["modelDigest"]);
    let moved = text(
        &k,
        Verb::Sink,
        "urn:bayes:question:train",
        &[
            ("content", "lunch on tuesday"),
            ("label", "yes"),
            ("as", "application/json"),
        ],
    );
    let moved: Value = serde_json::from_str(&moved).unwrap();
    assert_eq!(moved["previous"], "no");
    assert_eq!(moved["label"], "yes");
    let removed = text(
        &k,
        Verb::Delete,
        "urn:bayes:question:untrain",
        &[("content", "lunch on tuesday"), ("as", "application/json")],
    );
    let removed: Value = serde_json::from_str(&removed).unwrap();
    assert_eq!(removed["removed"], "yes");
    assert_eq!(removed["messages"], serde_json::json!({"no": 0, "yes": 1}));
}

#[test]
fn labels_and_digest_survive_reopen_and_cannot_be_reinterpreted() {
    let s = Scratch::new();
    let config = || ModelConfig::new("question").labels("yes", "no").unwrap();
    let before = {
        let bayes = Bayes::open(s.path(), [config()]).unwrap();
        let k = kernel(&bayes);
        train(&k, "question", "announcement", "yes");
        json(&k, "question", "stats", &[])
    };
    {
        let bayes = Bayes::open(s.path(), [config().cutoffs(0.3, 0.8).unwrap()]).unwrap();
        let stats = json(&kernel(&bayes), "question", "stats", &[]);
        assert_eq!(stats["modelDigest"], before["modelDigest"]);
        assert_eq!(stats["messages"], before["messages"]);
    }
    let err = Bayes::open(s.path(), [ModelConfig::new("question")])
        .unwrap_err()
        .to_string();
    assert!(err.contains("persisted labels"), "{err}");
    for (high, low) in [
        ("yes", "yes"),
        ("unsure", "no"),
        ("", "no"),
        ("Yes", "no"),
        ("yes", "no\n"),
    ] {
        assert!(ModelConfig::new("question").labels(high, low).is_err());
    }
}

#[test]
fn canonical_digest_ignores_training_order_and_default_file_bytes_are_unchanged() {
    use ikigai_bayes::model::{Label, Model};
    use ikigai_bayes::Message;
    use std::collections::BTreeSet;
    let first = Message::new("first").digest();
    let second = Message::new("second").digest();
    let tokens = BTreeSet::from(["first".to_string()]);
    let mut a = Model::new("a");
    let mut b = Model::new("b");
    let legacy =
        br#"{"format":"ikigai-bayes/model","version":1,"model":"a","tokenizer":"1","messages":[]}"#;
    assert_eq!(a.to_bytes(), legacy);
    assert_eq!(Model::from_bytes("a", legacy).unwrap().to_bytes(), legacy);
    a.train(first, Label::Spam, &tokens).unwrap();
    a.train(second, Label::Ham, &BTreeSet::new()).unwrap();
    b.train(second, Label::Ham, &BTreeSet::new()).unwrap();
    b.train(first, Label::Spam, &tokens).unwrap();
    assert_eq!(a.digest(), b.digest());
    assert_eq!(
        a.digest(),
        Model::from_bytes("a", &a.to_bytes()).unwrap().digest()
    );
    let before = a.digest();
    assert_eq!(a.untrain(&Message::new("absent").digest()), None);
    assert_eq!(a.digest(), before);
}

#[test]
fn failed_moves_and_untraining_preserve_the_state_digest() {
    let s = Scratch::new();
    let bayes = Bayes::open(s.path(), [ModelConfig::new("question")]).unwrap();
    let k = kernel(&bayes);
    train(&k, "question", "announcement", "spam");
    let before = json(&k, "question", "stats", &[]);
    std::fs::create_dir(s.path().join("question.json.tmp")).unwrap();
    assert!(issue(
        &k,
        Verb::Sink,
        "urn:bayes:question:train",
        &[("content", "announcement"), ("label", "ham")]
    )
    .is_err());
    assert_eq!(json(&k, "question", "stats", &[]), before);
    assert!(issue(
        &k,
        Verb::Delete,
        "urn:bayes:question:untrain",
        &[("content", "announcement")]
    )
    .is_err());
    assert_eq!(json(&k, "question", "stats", &[]), before);
    assert_eq!(
        json(&k, "question", "score", &[("content", "announcement")])["modelDigest"],
        before["modelDigest"]
    );
    std::fs::remove_dir(s.path().join("question.json.tmp")).unwrap();
}
