//! The resources as a host sees them: caching and the golden threads (measured with
//! `Kernel::is_cached`, not read off declarations), capabilities, idempotent training, exact
//! untraining, and the bounds.

mod common;

use common::*;
use ikigai_bayes::{Bayes, ModelConfig, MAX_MESSAGE_BYTES};
use ikigai_core::{Capability, Error, Verb};

const SPAM: &str = "cheap pills, buy now at https://pills.example.com/deal";
const HAM: &str = "can we meet on tuesday to review the draft";

#[test]
fn a_score_is_cached_until_the_model_learns_then_recomputes() {
    let bayes = two_models();
    let k = kernel(&bayes);
    let probe = request(
        Verb::Source,
        "urn:bayes:contact:score",
        &[("content", "cheap pills")],
    );
    let root = Capability::root();

    let first = score(&k, "contact", "cheap pills");
    assert!(first.starts_with("unsure 0.500000\n"), "{first}");
    assert!(
        k.is_cached(&probe, &root),
        "a score is cached on its inputs"
    );
    assert_eq!(score(&k, "contact", "cheap pills"), first);

    train(&k, "contact", SPAM, "spam");
    assert!(
        !k.is_cached(&probe, &root),
        "training cuts the model's thread, so the cached score is gone"
    );
    let second = score(&k, "contact", "cheap pills");
    assert!(second.starts_with("spam "), "{second}");
    assert!(
        k.is_cached(&probe, &root),
        "and the new score is cached in turn"
    );
}

#[test]
fn untraining_cuts_the_score_too() {
    let bayes = two_models();
    let k = kernel(&bayes);
    let probe = request(Verb::Source, "urn:bayes:contact:score", &[("content", "x")]);
    train(&k, "contact", SPAM, "spam");
    score(&k, "contact", "x");
    assert!(k.is_cached(&probe, &Capability::root()));
    text(
        &k,
        Verb::Delete,
        "urn:bayes:contact:untrain",
        &[("content", SPAM)],
    );
    assert!(!k.is_cached(&probe, &Capability::root()));
}

#[test]
fn another_models_training_cuts_nothing_here() {
    let bayes = two_models();
    let k = kernel(&bayes);
    let probe = request(Verb::Source, "urn:bayes:contact:score", &[("content", "x")]);
    let stats = request(Verb::Source, "urn:bayes:contact:stats", &[]);
    score(&k, "contact", "x");
    text(&k, Verb::Source, "urn:bayes:contact:stats", &[]);
    train(&k, "other", SPAM, "spam");
    assert!(k.is_cached(&probe, &Capability::root()));
    assert!(k.is_cached(&stats, &Capability::root()));
    // And `other` learned: `contact` did not.
    assert!(text(&k, Verb::Source, "urn:bayes:contact:stats", &[]).contains("spam 0 ham 0"));
    assert!(text(&k, Verb::Source, "urn:bayes:other:stats", &[]).contains("spam 1 ham 0"));
}

#[test]
fn stats_recompute_after_training() {
    let bayes = two_models();
    let k = kernel(&bayes);
    let probe = request(Verb::Source, "urn:bayes:contact:stats", &[]);
    let before = text(&k, Verb::Source, "urn:bayes:contact:stats", &[]);
    assert!(k.is_cached(&probe, &Capability::root()));
    train(&k, "contact", HAM, "ham");
    assert!(!k.is_cached(&probe, &Capability::root()));
    let after = text(&k, Verb::Source, "urn:bayes:contact:stats", &[]);
    assert_ne!(before, after);
    assert_eq!(
        after,
        "model contact\nmessages spam 0 ham 1\ntokens 9\ncutoffs ham 0.2 spam 0.9\ntokenizer 1\nmodelDigest b3:2f2ce99c9621517a5b3b779e96b5b4d821ba4790ff5e88e1410dcd942a01ed16\n"
    );
}

#[test]
fn each_model_enforces_its_own_capabilities_exactly() {
    let bayes = two_models();
    let k = kernel(&bayes);
    let reader = Capability::scoped(["urn:cap:bayes:read:contact"]);
    let trainer = Capability::scoped(["urn:cap:bayes:train:contact"]);
    let other_reader = Capability::scoped(["urn:cap:bayes:read:other"]);
    let none = Capability::scoped(Vec::<String>::new());
    let score_args = [("content", "x")];
    let train_args = [("content", "x"), ("label", "spam")];

    let ok = |cap: &Capability, verb, iri: &str, args: &[(&str, &str)]| {
        issue_as(&k, cap, verb, iri, args).is_ok()
    };
    let denied = |cap: &Capability, verb, iri: &str, args: &[(&str, &str)]| {
        matches!(issue_as(&k, cap, verb, iri, args), Err(Error::Denied(_)))
    };

    assert!(ok(
        &reader,
        Verb::Source,
        "urn:bayes:contact:score",
        &score_args
    ));
    assert!(ok(&reader, Verb::Source, "urn:bayes:contact:stats", &[]));
    assert!(denied(
        &reader,
        Verb::Sink,
        "urn:bayes:contact:train",
        &train_args
    ));
    assert!(denied(
        &reader,
        Verb::Delete,
        "urn:bayes:contact:untrain",
        &score_args
    ));

    assert!(ok(
        &trainer,
        Verb::Sink,
        "urn:bayes:contact:train",
        &train_args
    ));
    assert!(ok(
        &trainer,
        Verb::Delete,
        "urn:bayes:contact:untrain",
        &score_args
    ));
    assert!(denied(
        &trainer,
        Verb::Source,
        "urn:bayes:contact:score",
        &score_args
    ));

    // A grant for one model reaches no other.
    assert!(denied(
        &other_reader,
        Verb::Source,
        "urn:bayes:contact:score",
        &score_args
    ));
    assert!(ok(
        &other_reader,
        Verb::Source,
        "urn:bayes:other:score",
        &score_args
    ));

    // The list of models is public: names only, and the capability each part requires.
    assert!(ok(&none, Verb::Source, "urn:bayes:models", &[]));
}

#[test]
fn training_is_idempotent_and_a_relabel_moves() {
    let bayes = two_models();
    let k = kernel(&bayes);
    let first = train(&k, "contact", SPAM, "spam");
    assert!(first.starts_with("added b3:"), "{first}");
    assert!(first.ends_with("messages spam 1 ham 0\n"), "{first}");
    let again = train(&k, "contact", SPAM, "spam");
    assert!(again.starts_with("unchanged b3:"), "{again}");
    assert!(again.ends_with("messages spam 1 ham 0\n"), "{again}");
    let moved = train(&k, "contact", SPAM, "ham");
    assert!(moved.starts_with("moved b3:"), "{moved}");
    assert!(moved.contains(" from spam to ham\n"), "{moved}");
    assert!(moved.ends_with("messages spam 0 ham 1\n"), "{moved}");
}

#[test]
fn untrain_by_digest_restores_the_exact_prior_state() {
    let bayes = two_models();
    let k = kernel(&bayes);
    train(&k, "contact", HAM, "ham");
    let stats_before = text(&k, Verb::Source, "urn:bayes:contact:stats", &[]);
    let score_before = score(&k, "contact", "cheap pills tuesday");

    let added = train(&k, "contact", SPAM, "spam");
    let digest = added
        .split_whitespace()
        .find(|w| w.starts_with("b3:"))
        .unwrap()
        .to_string();
    assert_ne!(score(&k, "contact", "cheap pills tuesday"), score_before);

    let removed = text(
        &k,
        Verb::Delete,
        "urn:bayes:contact:untrain",
        &[("digest", &digest)],
    );
    assert_eq!(
        removed,
        format!("removed {digest} (was spam)\nmessages spam 0 ham 1\n")
    );
    assert_eq!(
        text(&k, Verb::Source, "urn:bayes:contact:stats", &[]),
        stats_before
    );
    assert_eq!(score(&k, "contact", "cheap pills tuesday"), score_before);

    // Gone is gone: a second untrain is NotFound.
    assert!(matches!(
        issue(
            &k,
            Verb::Delete,
            "urn:bayes:contact:untrain",
            &[("digest", &digest)]
        ),
        Err(Error::NotFound(_))
    ));
}

#[test]
fn fields_change_the_digest_and_the_tokens() {
    let bayes = two_models();
    let k = kernel(&bayes);
    text(
        &k,
        Verb::Sink,
        "urn:bayes:contact:train",
        &[
            ("content", "hello"),
            ("email", "someone@spam-domain.example"),
            ("label", "spam"),
        ],
    );
    // The same body with no field is a different message, and its tokens hold no field.
    let plain = score(&k, "contact", "hello");
    let with_field = text(
        &k,
        Verb::Source,
        "urn:bayes:contact:score",
        &[("content", "hi"), ("email", "other@spam-domain.example")],
    );
    assert!(
        with_field.contains(" email:spam-domain.example\n"),
        "{with_field}"
    );
    assert!(!plain.contains("email:"), "{plain}");
    // Untraining by content needs the same fields to find it.
    assert!(matches!(
        issue(
            &k,
            Verb::Delete,
            "urn:bayes:contact:untrain",
            &[("content", "hello")]
        ),
        Err(Error::NotFound(_))
    ));
    assert!(issue(
        &k,
        Verb::Delete,
        "urn:bayes:contact:untrain",
        &[
            ("content", "hello"),
            ("email", "someone@spam-domain.example")
        ],
    )
    .is_ok());
}

/// The spam message's words and its link, nothing the ham message holds.
const LINKED: &str = "cheap pills https://pills.example.com/other";

#[test]
fn the_json_and_turtle_faces() {
    let bayes = two_models();
    let k = kernel(&bayes);
    train(&k, "contact", SPAM, "spam");
    train(&k, "contact", HAM, "ham");
    let json: serde_json::Value = serde_json::from_slice(
        &issue(
            &k,
            Verb::Source,
            "urn:bayes:contact:score",
            &[("content", LINKED), ("as", "application/json")],
        )
        .unwrap()
        .bytes,
    )
    .unwrap();
    assert_eq!(json["class"], "spam");
    assert_eq!(json["model"], "contact");
    // Six tokens seen only in spam, all equally strong, so ordered by token.
    assert_eq!(json["clueCount"], 6);
    assert_eq!(json["clues"][0]["token"], "cheap");
    assert_eq!(json["clues"][1]["token"], "links:1");
    assert_eq!(json["cutoffs"]["spam"], 0.9);

    let rep = issue(
        &k,
        Verb::Source,
        "urn:bayes:contact:score",
        &[("content", LINKED), ("as", "text/turtle")],
    )
    .unwrap();
    assert_eq!(rep.repr_type.media_type, "text/turtle");
    let ttl = String::from_utf8(rep.bytes).unwrap();
    assert!(ttl.contains("a schema:Rating"), "{ttl}");
    assert!(ttl.contains("schema:category \"spam\""), "{ttl}");
    assert!(!ttl.contains("_:"), "no blank nodes: {ttl}");
    assert!(ttl.contains(":clue:cheap>"), "{ttl}");
    assert!(ttl.contains(":clue:url-domain%3Aexample.com>"), "{ttl}");
}

#[test]
fn the_models_list() {
    let bayes = two_models();
    let k = kernel(&bayes);
    assert_eq!(
        text(&k, Verb::Source, "urn:bayes:models", &[]),
        "contact\nother"
    );
    let json: serde_json::Value = serde_json::from_slice(
        &issue(
            &k,
            Verb::Source,
            "urn:bayes:models",
            &[("as", "application/json")],
        )
        .unwrap()
        .bytes,
    )
    .unwrap();
    assert_eq!(
        json["models"][0]["readCapability"],
        "urn:cap:bayes:read:contact"
    );
    assert_eq!(
        json["models"][1]["trainCapability"],
        "urn:cap:bayes:train:other"
    );
}

#[test]
fn bad_calls_are_refused_typed() {
    let bayes = two_models();
    let k = kernel(&bayes);
    let invalid = |r: ikigai_core::Result<_>, arg: &str| matches!(r, Err(Error::InvalidArgument { ref name, .. }) if name == arg);
    let big = "a ".repeat(MAX_MESSAGE_BYTES / 2 + 1);
    assert!(invalid(
        issue(
            &k,
            Verb::Source,
            "urn:bayes:contact:score",
            &[("content", &big)]
        ),
        "content"
    ));
    // The bound counts the fields too.
    let half = "a".repeat(MAX_MESSAGE_BYTES / 2 + 1);
    assert!(invalid(
        issue(
            &k,
            Verb::Source,
            "urn:bayes:contact:score",
            &[("content", &half), ("subject", &half)]
        ),
        "content"
    ));
    assert!(invalid(
        issue(
            &k,
            Verb::Source,
            "urn:bayes:contact:score",
            &[("content", "x"), ("clues", "151")]
        ),
        "clues"
    ));
    assert!(invalid(
        issue(
            &k,
            Verb::Source,
            "urn:bayes:contact:score",
            &[("content", "x"), ("as", "text/html")]
        ),
        "as"
    ));
    assert!(invalid(
        issue(
            &k,
            Verb::Sink,
            "urn:bayes:contact:train",
            &[("content", "x"), ("label", "eggs")]
        ),
        "label"
    ));
    assert!(matches!(
        issue(&k, Verb::Sink, "urn:bayes:contact:train", &[("content", "x")]),
        Err(Error::MissingArgument(ref n)) if n == "label"
    ));
    assert!(matches!(
        issue(&k, Verb::Delete, "urn:bayes:contact:untrain", &[]),
        Err(Error::MissingArgument(ref n)) if n == "content"
    ));
    assert!(invalid(
        issue(
            &k,
            Verb::Delete,
            "urn:bayes:contact:untrain",
            &[("content", "x"), ("digest", "b3:00")]
        ),
        "digest"
    ));
    assert!(invalid(
        issue(
            &k,
            Verb::Delete,
            "urn:bayes:contact:untrain",
            &[("digest", "md5:abc")]
        ),
        "digest"
    ));
}

#[test]
fn a_bad_model_configuration_is_refused_at_open() {
    let err = |models: Vec<ModelConfig>| Bayes::in_memory(models).unwrap_err().to_string();
    assert!(err(vec![ModelConfig::new("Bad")]).contains("not a model name"));
    assert!(err(vec![ModelConfig::new("models")]).contains("not a model name"));
    assert!(err(vec![ModelConfig::new("a"), ModelConfig::new("a")]).contains("twice"));
    let many: Vec<ModelConfig> = (0..=ikigai_bayes::MAX_MODELS)
        .map(|i| ModelConfig::new(format!("m{i}")))
        .collect();
    assert!(err(many).contains("more than 64 models"));
    assert!(ModelConfig::new("a").cutoffs(0.9, 0.1).is_err());
}

#[test]
fn per_model_cutoffs_classify() {
    let bayes = Bayes::in_memory([
        ModelConfig::new("strict").cutoffs(0.5, 0.5).unwrap(),
        ModelConfig::new("default"),
    ])
    .unwrap();
    let k = kernel(&bayes);
    // An untrained model scores 0.5: unsure by default, spam at a 0.5 spam cutoff.
    assert!(score(&k, "default", "anything").starts_with("unsure 0.500000"));
    assert!(score(&k, "strict", "anything").starts_with("spam 0.500000"));
    assert!(text(&k, Verb::Source, "urn:bayes:strict:stats", &[])
        .contains("cutoffs ham 0.5 spam 0.5\n"));
}
