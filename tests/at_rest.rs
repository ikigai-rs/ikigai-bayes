//! A model's state, sealed at rest: on disk is age ciphertext holding no token in plaintext, a
//! sealed model answers exactly what the unsealed one did, plaintext is refused until the host
//! reseals it, and a key that lacks the model's is refused rather than starting it empty.

mod common;

use common::*;
use ikigai_bayes::at_rest::{Migration, StateKey};
use ikigai_bayes::{Bayes, Message, ModelConfig};
use ikigai_core::Verb;
use ikigai_encrypt::age::{ExposeSecret, Identity};

const MAGIC: &[u8] = b"age-encryption.org/v1\n";

/// Messages whose every token is long enough that finding it in ciphertext by chance is
/// negligible, so a token found on disk is a token kept in the clear.
const SPAM: [&str; 3] = [
    "zanzibarquokka pharmacy discount, purchase immediately",
    "unbelievable lottery winnings waiting, transfer processing",
    "pharmacy discount unbelievable, purchase quokkaburger",
];
const HAM: [&str; 3] = [
    "quarterly planning meeting tuesday afternoon",
    "invoice attached regarding consulting engagement",
    "afternoon meeting rescheduled, conference upstairs",
];

fn key_text() -> String {
    format!(
        "# created by age-keygen\n{}\n",
        Identity::generate().to_string().expose_secret()
    )
}

fn key() -> StateKey {
    StateKey::parse(&key_text()).unwrap()
}

fn contact() -> [ModelConfig; 1] {
    [ModelConfig::new("contact")]
}

fn teach(bayes: &Bayes) {
    let k = kernel(bayes);
    for m in SPAM {
        train(&k, "contact", m, "spam");
    }
    for m in HAM {
        train(&k, "contact", m, "ham");
    }
}

/// Every answer a host reads from the model: each face of a score, and the stats.
fn answers(bayes: &Bayes) -> Vec<String> {
    let k = kernel(bayes);
    let mut out = vec![text(&k, Verb::Source, "urn:bayes:contact:stats", &[])];
    for probe in [
        "pharmacy discount on tuesday",
        "zanzibarquokka meeting afternoon",
        "invoice for consulting",
        "never seen words",
    ] {
        for face in ["text/plain", "application/json"] {
            out.push(text(
                &k,
                Verb::Source,
                "urn:bayes:contact:score",
                &[("content", probe), ("as", face)],
            ));
        }
    }
    out
}

/// Every token of every trained message that is found in any file under `dir`.
fn tokens_on_disk(dir: &std::path::Path) -> Vec<String> {
    let files: Vec<Vec<u8>> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| std::fs::read(e.unwrap().path()).unwrap())
        .collect();
    let mut found = Vec::new();
    for m in SPAM.iter().chain(&HAM) {
        for token in Message::new(*m).tokens().unwrap() {
            if token.len() >= 6
                && files
                    .iter()
                    .any(|f| f.windows(token.len()).any(|w| w == token.as_bytes()))
            {
                found.push(token);
            }
        }
    }
    found
}

#[test]
fn state_on_disk_carries_no_plaintext_token() {
    let s = Scratch::new();
    let key = key();
    {
        let bayes = Bayes::open_sealed(s.path(), &key, contact()).unwrap();
        teach(&bayes);
    }
    let model = std::fs::read(s.path().join("contact.json")).unwrap();
    assert!(model.starts_with(MAGIC), "the model is one age file");
    assert_eq!(tokens_on_disk(s.path()), Vec::<String>::new());

    // The same check finds them in a plaintext state, so it would catch a leak.
    let plain = Scratch::new();
    teach(&Bayes::open(plain.path(), contact()).unwrap());
    assert!(tokens_on_disk(plain.path()).contains(&"zanzibarquokka".to_string()));
}

#[test]
fn a_sealed_model_scores_exactly_as_the_unsealed_one_did() {
    let s = Scratch::new();
    let key = key();
    let before = {
        let bayes = Bayes::open(s.path(), contact()).unwrap();
        teach(&bayes);
        answers(&bayes)
    };
    assert!(!tokens_on_disk(s.path()).is_empty(), "written in the clear");

    let resealed = Bayes::open_sealed_with(s.path(), &key, Migration::Reseal, contact()).unwrap();
    assert_eq!(answers(&resealed), before);
    drop(resealed);
    assert!(std::fs::read(s.path().join("contact.json"))
        .unwrap()
        .starts_with(MAGIC));
    assert_eq!(tokens_on_disk(s.path()), Vec::<String>::new());

    // Read back from the ciphertext alone, it still answers the same, and keeps learning.
    let sealed = Bayes::open_sealed(s.path(), &key, contact()).unwrap();
    assert_eq!(answers(&sealed), before);
    train(&kernel(&sealed), "contact", "unbelievable transfer", "spam");
    drop(sealed);
    assert_eq!(tokens_on_disk(s.path()), Vec::<String>::new());
    let reopened = Bayes::open_sealed(s.path(), &key, contact()).unwrap();
    assert!(answers(&reopened)[0].contains("spam 4 ham 3"));
}

#[test]
fn a_plaintext_model_is_refused_until_it_is_resealed() {
    let s = Scratch::new();
    drop(Bayes::open(s.path(), contact()).unwrap());
    let err = Bayes::open_sealed(s.path(), &key(), contact())
        .unwrap_err()
        .to_string();
    assert!(err.contains("model `contact` is not sealed"), "{err}");
    assert!(err.contains("Migration::Reseal"), "{err}");
    // Refused, not rewritten.
    assert!(std::fs::read(s.path().join("contact.json"))
        .unwrap()
        .starts_with(b"{"));
}

#[test]
fn a_reseal_seals_every_registered_model_not_only_those_offered() {
    let s = Scratch::new();
    let key = key();
    drop(
        Bayes::open(
            s.path(),
            [ModelConfig::new("contact"), ModelConfig::new("retired")],
        )
        .unwrap(),
    );
    drop(Bayes::open_sealed_with(s.path(), &key, Migration::Reseal, contact()).unwrap());
    for name in ["contact.json", "retired.json"] {
        let bytes = std::fs::read(s.path().join(name)).unwrap();
        assert!(bytes.starts_with(MAGIC), "{name} is sealed");
    }
    assert!(Bayes::open_sealed(s.path(), &key, [ModelConfig::new("retired")]).is_ok());
}

#[test]
fn a_key_that_lacks_the_models_key_is_refused_not_started_empty() {
    let s = Scratch::new();
    teach(&Bayes::open_sealed(s.path(), &key(), contact()).unwrap());
    let err = Bayes::open_sealed(s.path(), &key(), contact())
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("model `contact` is sealed to a key the state identity does not hold"),
        "{err}"
    );
}

#[test]
fn a_rotation_reseals_to_the_new_identity() {
    let s = Scratch::new();
    let old = key_text();
    let new = key_text();
    let before = {
        let bayes =
            Bayes::open_sealed(s.path(), &StateKey::parse(&old).unwrap(), contact()).unwrap();
        teach(&bayes);
        answers(&bayes)
    };
    // The new identity first, the old one after it, opens what the old one sealed.
    let both = StateKey::parse(&format!("{new}{old}")).unwrap();
    drop(Bayes::open_sealed(s.path(), &both, contact()).unwrap());
    let new_only = StateKey::parse(&new).unwrap();
    assert!(
        Bayes::open_sealed(s.path(), &new_only, contact()).is_err(),
        "not yet resealed"
    );
    drop(Bayes::open_sealed_with(s.path(), &both, Migration::Reseal, contact()).unwrap());
    let bayes = Bayes::open_sealed(s.path(), &new_only, contact()).unwrap();
    assert_eq!(answers(&bayes), before);
}

#[test]
fn a_bad_key_is_refused_by_name_without_echoing_it() {
    let err = StateKey::parse("# nothing\n").unwrap_err().to_string();
    assert!(err.contains("holds no age identity"), "{err}");
    let err = StateKey::parse("AGE-SECRET-KEY-1NOTAKEY\n")
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("line 1 is not an X25519 age identity"),
        "{err}"
    );
    assert!(!err.contains("NOTAKEY"), "{err}");

    let s = Scratch::new();
    let missing = StateKey::load(s.path().join("absent.key"))
        .unwrap_err()
        .to_string();
    assert!(
        missing.contains("absent.key") && missing.contains("does not exist"),
        "{missing}"
    );
}

#[cfg(unix)]
#[test]
fn a_key_file_another_account_can_read_is_refused() {
    use std::os::unix::fs::PermissionsExt;

    let s = Scratch::new();
    let dir = s.path().join("keys");
    std::fs::create_dir(&dir).unwrap();
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).unwrap();
    let path = dir.join("state.key");
    std::fs::write(&path, key_text()).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
    let err = StateKey::load(&path).unwrap_err().to_string();
    assert!(err.contains("has mode 0644"), "{err}");
    assert!(
        err.contains(&format!("`chmod 600 {}`", path.display())),
        "{err}"
    );

    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
    let key = StateKey::load(&path).unwrap();
    assert!(key.recipient().starts_with("age1"));

    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o777)).unwrap();
    let err = StateKey::load(&path).unwrap_err().to_string();
    assert!(err.contains("`chmod go-w"), "{err}");
}

#[cfg(unix)]
#[test]
fn the_state_directory_is_closed_to_other_accounts() {
    use std::os::unix::fs::PermissionsExt;

    let s = Scratch::new();
    std::fs::set_permissions(s.path(), std::fs::Permissions::from_mode(0o755)).unwrap();
    drop(Bayes::open_sealed(s.path(), &key(), contact()).unwrap());
    let mode = std::fs::metadata(s.path()).unwrap().permissions().mode() & 0o7777;
    assert_eq!(mode, 0o700);
}
