#[cfg(not(target_family = "wasm"))]
#[test]
fn failed_first_save_does_not_leave_model_bytes() {
    use ikigai_bayes::{
        model::Model,
        store::{DirStore, Store},
    };
    let dir = std::env::current_dir()
        .unwrap()
        .join("target")
        .join(format!("audit-store-{}", std::process::id()));
    let store = DirStore::open(&dir).unwrap();
    std::fs::create_dir(dir.join("models")).unwrap();
    let result = store.save("contact", &Model::new("contact").to_bytes());
    let exists = dir.join("contact.json").exists();
    drop(store);
    std::fs::remove_dir_all(&dir).unwrap();
    assert!(result.is_err());
    assert!(!exists);
}

#[test]
fn successful_direct_training_round_trips() {
    use ikigai_bayes::model::{Label, Model, MAX_TOKEN_BYTES};
    use ikigai_core::ContentId;
    use std::collections::BTreeSet;

    let mut model = Model::new("m");
    let tokens = BTreeSet::from(["x".repeat(MAX_TOKEN_BYTES + 1)]);
    if model
        .train(ContentId::of(b"message"), Label::Ham, &tokens)
        .is_ok()
    {
        assert!(Model::from_bytes("m", &model.to_bytes()).is_ok());
    }
}

#[test]
fn direct_training_refuses_invalid_new_tokens_without_mutation() {
    use ikigai_bayes::{
        model::{Change, Label, Model, Refusal, MAX_TOKEN_BYTES},
        tokenize::MAX_TOKENS,
    };
    use ikigai_core::ContentId;
    use std::collections::BTreeSet;

    let mut model = Model::new("m");
    let existing = ContentId::of(b"existing");
    let valid = BTreeSet::from(["x".repeat(MAX_TOKEN_BYTES)]);
    assert_eq!(model.train(existing, Label::Ham, &valid), Ok(Change::Added));
    let before = model.to_bytes();
    let digest = model.digest();
    let too_long = BTreeSet::from(["é".repeat(MAX_TOKEN_BYTES / 2 + 1)]);
    assert_eq!(
        model.train(ContentId::of(b"long"), Label::Spam, &too_long),
        Err(Refusal::TokenTooLong {
            bytes: MAX_TOKEN_BYTES + 2
        })
    );
    let too_many: BTreeSet<_> = (0..=MAX_TOKENS).map(|i| i.to_string()).collect();
    assert_eq!(
        model.train(ContentId::of(b"many"), Label::Spam, &too_many),
        Err(Refusal::TooManyMessageTokens {
            count: MAX_TOKENS + 1
        })
    );
    assert_eq!(model.to_bytes(), before);
    assert_eq!(model.digest(), digest);
    assert_eq!(model.nham(), 1);
    assert_eq!(model.nspam(), 0);
    assert_eq!(model.counts(&"x".repeat(MAX_TOKEN_BYTES)), (0, 1));
    assert_eq!(
        model.train(existing, Label::Ham, &too_long),
        Ok(Change::Unchanged)
    );
    assert_eq!(
        model.train(existing, Label::Spam, &too_many),
        Ok(Change::Moved { from: Label::Ham })
    );
    assert!(Model::from_bytes("m", &model.to_bytes()).is_ok());

    let mut boundary = Model::new("m");
    let exactly: BTreeSet<_> = (0..MAX_TOKENS).map(|i| i.to_string()).collect();
    assert_eq!(
        boundary.train(ContentId::of(b"boundary"), Label::Ham, &exactly),
        Ok(Change::Added)
    );
    assert!(Model::from_bytes("m", &boundary.to_bytes()).is_ok());
}
