//! Synthetic edge cases for the scoring, tokenization, and concurrent snapshot contracts.

mod common;

use std::collections::BTreeMap;

use common::*;
use ikigai_bayes::classify::Settings;
use ikigai_bayes::model::{Label, Model};
use ikigai_bayes::{Message, MAX_MESSAGE_BYTES};
use ikigai_core::{Error, Verb};
use serde_json::Value;

fn json_score(k: &ikigai_core::Kernel, content: &str) -> Value {
    serde_json::from_str(&text(
        k,
        Verb::Source,
        "urn:bayes:contact:score",
        &[("content", content), ("as", "application/json")],
    ))
    .unwrap()
}

#[test]
fn scores_are_finite_for_empty_single_token_and_bounded_large_messages() {
    let bayes = two_models();
    let k = kernel(&bayes);
    let large = "界".repeat(MAX_MESSAGE_BYTES / 3);
    for label in [None, Some("spam"), Some("ham")] {
        if let Some(label) = label {
            train(&k, "contact", "synthetic training words", label);
        }
        for content in ["", "synthetic", large.as_str()] {
            let score = json_score(&k, content);
            let p = score["score"].as_f64().unwrap();
            assert!(p.is_finite() && (0.0..=1.0).contains(&p), "{score}");
        }
    }
    let megabyte = "a".repeat(1024 * 1024);
    for verb in [Verb::Source, Verb::Sink] {
        let part = if verb == Verb::Source {
            "score"
        } else {
            "train"
        };
        let err = issue(
            &k,
            verb,
            &format!("urn:bayes:contact:{part}"),
            &[("content", &megabyte), ("label", "spam")],
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidArgument { ref name, .. } if name == "content"));
        assert!(err.to_string().contains("refused, not truncated"));
    }
}

#[test]
fn unseen_tokens_do_not_change_a_default_score() {
    let mut model = Model::new("contact");
    let message = Message::new("synthetic training words");
    let mut tokens = message.tokens().unwrap();
    model.train(message.digest(), Label::Spam, &tokens).unwrap();
    let before = model.score(&tokens, &Settings::default());
    tokens.insert("never-observed".into());
    assert_eq!(model.score(&tokens, &Settings::default()), before);
    model.untrain(&message.digest());
    assert_eq!(model.to_bytes(), Model::new("contact").to_bytes());
}

#[test]
fn unicode_numbers_and_long_words_follow_the_documented_token_rules() {
    let body = "ÉCOLE İSTANBUL cafe\u{301} 東京大学 12345";
    let message = Message::new(body);
    let tokens = message.tokens().unwrap();
    for token in [
        "école",
        "i\u{307}stanbul",
        "cafe\u{301}",
        "東京大学",
        "12345",
    ] {
        assert!(tokens.contains(token), "missing {token}: {tokens:?}");
    }
    // Unicode lowercasing is documented; normalization and CJK segmentation are not.
    assert!(!tokens.contains("café"));
    let long = Message::new("界".repeat(31)).tokens().unwrap();
    assert!(long.contains("skip:界 30"));
    let mut model = Model::new("contact");
    model.train(message.digest(), Label::Spam, &tokens).unwrap();
    let loaded = Model::from_bytes("contact", &model.to_bytes()).unwrap();
    assert_eq!(
        loaded.score(&message.tokens().unwrap(), &Settings::default()),
        model.score(&tokens, &Settings::default())
    );
}

#[test]
fn concurrent_corrections_and_scores_observe_complete_model_snapshots() {
    let bayes = two_models();
    let k = kernel(&bayes);
    let content = "synthetic snapshot words";
    let empty = json_score(&k, content);
    train(&k, "contact", content, "spam");
    let spam = json_score(&k, content);
    train(&k, "contact", content, "ham");
    let ham = json_score(&k, content);
    let known: BTreeMap<String, Value> = [empty, spam, ham.clone()]
        .into_iter()
        .map(|score| (score["modelDigest"].as_str().unwrap().to_string(), score))
        .collect();
    std::thread::scope(|scope| {
        scope.spawn(|| {
            for _ in 0..50 {
                train(&k, "contact", content, "spam");
                train(&k, "contact", content, "ham");
            }
        });
        scope.spawn(|| {
            for _ in 0..100 {
                let answer = json_score(&k, content);
                let digest = answer["modelDigest"].as_str().unwrap();
                assert_eq!(known.get(digest), Some(&answer));
            }
        });
    });
    assert_eq!(json_score(&k, content), ham);
}
