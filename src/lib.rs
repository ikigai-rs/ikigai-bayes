//! `ikigai-bayes` — a naive-Bayes text classifier as ikigai resources.
//!
//! One model per name, named by the HOST: the host opens a [`Bayes`] over a state directory it
//! chooses and the model names it offers, and mounts [`space`]. Each model is five resources:
//!
//! | resource | verb | answer | capability |
//! |---|---|---|---|
//! | `urn:bayes:{model}:score` | Source | the score, the class (`spam`/`ham`/`unsure`) and the strongest clues | `urn:cap:bayes:read:{model}` |
//! | `urn:bayes:{model}:train` | Sink | what changed (`added`, `moved`, `unchanged`) | `urn:cap:bayes:train:{model}` |
//! | `urn:bayes:{model}:untrain` | Delete | what was removed | `urn:cap:bayes:train:{model}` |
//! | `urn:bayes:{model}:stats` | Source | message and token counts, the cutoffs, the tokenizer version | `urn:cap:bayes:read:{model}` |
//! | `urn:bayes:models` | Source | the models this host offers | none |
//!
//! The method is SpamBayes's: Robinson's per-token probability with its `s`/`x` prior, and
//! Fisher's chi-squared combining of the most significant clues ([`classify`]). The tokenizer is
//! versioned and documented rule by rule ([`tokenize`](mod@tokenize)). A model is the set of messages it was
//! taught; its counts are derived from that set ([`model`]), which is what makes training
//! idempotent per message and untraining exact.
//!
//! # Caching and the golden threads
//!
//! A score is cacheable on its inputs, and hangs from the model's two golden threads,
//! `urn:bayes:{model}:train` and `urn:bayes:{model}:untrain`. The kernel cuts the thread named
//! after a write's target after every write through it, so a score is served from the cache until
//! that model learns or forgets something, and then recomputes. Another model's training cuts
//! nothing here.
//!
//! # Example
//!
//! A host opens the models it offers and mounts the space; here, in memory, with one model taught
//! one spam and one ham message.
//!
//! ```
//! use std::sync::Arc;
//! use futures::executor::block_on;
//! use ikigai_bayes::{Bayes, ModelConfig};
//! use ikigai_core::{ArgRef, Capability, Iri, Kernel, Request, Verb};
//!
//! let bayes = Bayes::in_memory([ModelConfig::new("contact")]).unwrap();
//! let kernel = Kernel::new(Arc::new(ikigai_bayes::space(&bayes)));
//! let call = |verb, iri: &str, args: &[(&str, &str)]| {
//!     let mut request = Request::new(verb, Iri::parse(iri).unwrap());
//!     for (name, value) in args {
//!         request = request.with_arg(*name, ArgRef::Inline(value.as_bytes().to_vec()));
//!     }
//!     let rep = block_on(kernel.issue(request, &Capability::root())).unwrap();
//!     String::from_utf8(rep.bytes).unwrap()
//! };
//!
//! let train = "urn:bayes:contact:train";
//! call(Verb::Sink, train, &[("content", "cheap pills, buy now"), ("label", "spam")]);
//! call(Verb::Sink, train, &[("content", "can we meet on tuesday"), ("label", "ham")]);
//!
//! // The text face: the class and the score, the message's digest, how many clues are shown of
//! // how many were combined, then one `<probability> <token>` line per clue, strongest first.
//! // Two clues of 0.844828 each combine to 0.920316.
//! let answer = call(Verb::Source, "urn:bayes:contact:score", &[("content", "cheap pills")]);
//! assert_eq!(
//!     answer,
//!     "spam 0.920316\n\
//!      digest b3:7d3bbbf2006ca64a1d7cd63f28099ea39765a2621e6187c4a1ad55c4a454aee4\n\
//!      clues 2 of 2\n\
//!      0.844828 cheap\n\
//!      0.844828 pills\n"
//! );
//! ```
//!
//! # The answers
//!
//! **Score**, `as=application/json`: the full-precision score, the class, the model's cutoffs,
//! how many clues were combined and the ones shown (`clues=`, default
//! [`DEFAULT_CLUES_SHOWN`]), and the tokenizer version.
//!
//! ```
//! # use std::sync::Arc;
//! # use futures::executor::block_on;
//! # use ikigai_bayes::{Bayes, ModelConfig};
//! # use ikigai_core::{ArgRef, Capability, Iri, Kernel, Request, Verb};
//! # let bayes = Bayes::in_memory([ModelConfig::new("contact")]).unwrap();
//! # let kernel = Kernel::new(Arc::new(ikigai_bayes::space(&bayes)));
//! # let call = |verb, iri: &str, args: &[(&str, &str)]| {
//! #     let mut request = Request::new(verb, Iri::parse(iri).unwrap());
//! #     for (name, value) in args {
//! #         request = request.with_arg(*name, ArgRef::Inline(value.as_bytes().to_vec()));
//! #     }
//! #     let rep = block_on(kernel.issue(request, &Capability::root())).unwrap();
//! #     String::from_utf8(rep.bytes).unwrap()
//! # };
//! # let train = "urn:bayes:contact:train";
//! # call(Verb::Sink, train, &[("content", "cheap pills, buy now"), ("label", "spam")]);
//! # call(Verb::Sink, train, &[("content", "can we meet on tuesday"), ("label", "ham")]);
//! let json = call(
//!     Verb::Source,
//!     "urn:bayes:contact:score",
//!     &[("content", "cheap pills"), ("as", "application/json")],
//! );
//! assert_eq!(
//!     json,
//!     r#"{"class":"spam","clueCount":2,"clues":[{"probability":0.8448275862068967,"token":"cheap"},{"probability":0.8448275862068967,"token":"pills"}],"cutoffs":{"ham":0.2,"spam":0.9},"digest":"b3:7d3bbbf2006ca64a1d7cd63f28099ea39765a2621e6187c4a1ad55c4a454aee4","model":"contact","score":0.920315826239094,"tokenizer":"1"}"#
//! );
//! ```
//!
//! **Score**, `as=text/turtle`: skolemized (no blank nodes), in schema.org, PROV and Dublin Core
//! terms. The score is `urn:bayes:{model}:score:{digest}`, a `schema:Rating` whose
//! `schema:category` is the class; each clue shown is a `schema:PropertyValue` at the score's IRI
//! plus `:clue:{percent-encoded token}`.
//!
//! ```
//! # use std::sync::Arc;
//! # use futures::executor::block_on;
//! # use ikigai_bayes::{Bayes, ModelConfig};
//! # use ikigai_core::{ArgRef, Capability, Iri, Kernel, Request, Verb};
//! # let bayes = Bayes::in_memory([ModelConfig::new("contact")]).unwrap();
//! # let kernel = Kernel::new(Arc::new(ikigai_bayes::space(&bayes)));
//! # let call = |verb, iri: &str, args: &[(&str, &str)]| {
//! #     let mut request = Request::new(verb, Iri::parse(iri).unwrap());
//! #     for (name, value) in args {
//! #         request = request.with_arg(*name, ArgRef::Inline(value.as_bytes().to_vec()));
//! #     }
//! #     let rep = block_on(kernel.issue(request, &Capability::root())).unwrap();
//! #     String::from_utf8(rep.bytes).unwrap()
//! # };
//! # let train = "urn:bayes:contact:train";
//! # call(Verb::Sink, train, &[("content", "cheap pills, buy now"), ("label", "spam")]);
//! # call(Verb::Sink, train, &[("content", "can we meet on tuesday"), ("label", "ham")]);
//! let ttl = call(
//!     Verb::Source,
//!     "urn:bayes:contact:score",
//!     &[("content", "cheap pills"), ("as", "text/turtle"), ("clues", "1")],
//! );
//! let score = "urn:bayes:contact:score:b3:7d3bbbf2006ca64a1d7cd63f28099ea39765a2621e6187c4a1ad55c4a454aee4";
//! assert_eq!(
//!     ttl,
//!     format!(
//!         "@prefix schema: <https://schema.org/> .\n\
//!          @prefix prov: <http://www.w3.org/ns/prov#> .\n\
//!          @prefix dcterms: <http://purl.org/dc/terms/> .\n\
//!          @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n\
//!          \n\
//!          <{score}> a schema:Rating ;\n  \
//!          schema:ratingValue \"0.920316\"^^xsd:decimal ;\n  \
//!          schema:worstRating \"0\"^^xsd:decimal ;\n  \
//!          schema:bestRating \"1\"^^xsd:decimal ;\n  \
//!          schema:category \"spam\" ;\n  \
//!          schema:ratingExplanation \"ham below 0.2, spam at or above 0.9; 2 clue(s) combined\" ;\n  \
//!          dcterms:identifier \"b3:7d3bbbf2006ca64a1d7cd63f28099ea39765a2621e6187c4a1ad55c4a454aee4\" ;\n  \
//!          prov:wasDerivedFrom <urn:bayes:contact:stats> ;\n  \
//!          schema:additionalProperty <{score}:clue:cheap> .\n\
//!          \n\
//!          <{score}:clue:cheap> a schema:PropertyValue ;\n  \
//!          schema:name \"cheap\" ;\n  \
//!          schema:value \"0.844828\"^^xsd:decimal .\n"
//!     )
//! );
//! ```
//!
//! **Train** answers what changed, then the model's message counts; **untrain** answers what it
//! removed. Both have a JSON face too.
//!
//! ```
//! # use std::sync::Arc;
//! # use futures::executor::block_on;
//! # use ikigai_bayes::{Bayes, ModelConfig};
//! # use ikigai_core::{ArgRef, Capability, Iri, Kernel, Request, Verb};
//! # let bayes = Bayes::in_memory([ModelConfig::new("contact")]).unwrap();
//! # let kernel = Kernel::new(Arc::new(ikigai_bayes::space(&bayes)));
//! # let call = |verb, iri: &str, args: &[(&str, &str)]| {
//! #     let mut request = Request::new(verb, Iri::parse(iri).unwrap());
//! #     for (name, value) in args {
//! #         request = request.with_arg(*name, ArgRef::Inline(value.as_bytes().to_vec()));
//! #     }
//! #     let rep = block_on(kernel.issue(request, &Capability::root())).unwrap();
//! #     String::from_utf8(rep.bytes).unwrap()
//! # };
//! # let train = "urn:bayes:contact:train";
//! # call(Verb::Sink, train, &[("content", "cheap pills, buy now"), ("label", "spam")]);
//! # call(Verb::Sink, train, &[("content", "can we meet on tuesday"), ("label", "ham")]);
//! let again = call(Verb::Sink, train, &[("content", "can we meet on tuesday"), ("label", "ham")]);
//! assert_eq!(
//!     again,
//!     "unchanged b3:8432362580eef64db2938e02109d97d62e6b82fe35d9db37a278c7eca7a5135a: already ham\n\
//!      messages spam 1 ham 1\n"
//! );
//! let moved = call(
//!     Verb::Sink,
//!     train,
//!     &[("content", "can we meet on tuesday"), ("label", "spam"), ("as", "application/json")],
//! );
//! assert_eq!(
//!     moved,
//!     r#"{"change":"moved","digest":"b3:8432362580eef64db2938e02109d97d62e6b82fe35d9db37a278c7eca7a5135a","label":"spam","messages":{"ham":0,"spam":2},"model":"contact","previous":"ham"}"#
//! );
//! let back = call(Verb::Sink, train, &[("content", "can we meet on tuesday"), ("label", "ham")]);
//! assert_eq!(
//!     back,
//!     "moved b3:8432362580eef64db2938e02109d97d62e6b82fe35d9db37a278c7eca7a5135a from spam to ham\n\
//!      messages spam 1 ham 1\n"
//! );
//! let removed = call(
//!     Verb::Delete,
//!     "urn:bayes:contact:untrain",
//!     &[("content", "cheap pills, buy now")],
//! );
//! assert_eq!(
//!     removed,
//!     "removed b3:3b1ba44f431be1efecab7244eea6623f45cb2c9f44b70c2cdcdd6573a2aa3bd1 (was spam)\n\
//!      messages spam 0 ham 1\n"
//! );
//! // A new message is `added b3:… as <label>`.
//! let added = call(Verb::Sink, train, &[("content", "cheap pills, buy now"), ("label", "spam")]);
//! assert!(added.starts_with("added b3:3b1ba44f"), "{added}");
//! ```
//!
//! **Stats** and **the models list** (one name per line by default; the JSON face names each
//! part and the capability it requires):
//!
//! ```
//! # use std::sync::Arc;
//! # use futures::executor::block_on;
//! # use ikigai_bayes::{Bayes, ModelConfig};
//! # use ikigai_core::{ArgRef, Capability, Iri, Kernel, Request, Verb};
//! # let bayes = Bayes::in_memory([ModelConfig::new("contact")]).unwrap();
//! # let kernel = Kernel::new(Arc::new(ikigai_bayes::space(&bayes)));
//! # let call = |verb, iri: &str, args: &[(&str, &str)]| {
//! #     let mut request = Request::new(verb, Iri::parse(iri).unwrap());
//! #     for (name, value) in args {
//! #         request = request.with_arg(*name, ArgRef::Inline(value.as_bytes().to_vec()));
//! #     }
//! #     let rep = block_on(kernel.issue(request, &Capability::root())).unwrap();
//! #     String::from_utf8(rep.bytes).unwrap()
//! # };
//! # let train = "urn:bayes:contact:train";
//! # call(Verb::Sink, train, &[("content", "cheap pills, buy now"), ("label", "spam")]);
//! # call(Verb::Sink, train, &[("content", "can we meet on tuesday"), ("label", "ham")]);
//! assert_eq!(
//!     call(Verb::Source, "urn:bayes:contact:stats", &[]),
//!     "model contact\nmessages spam 1 ham 1\ntokens 10\ncutoffs ham 0.2 spam 0.9\ntokenizer 1\n"
//! );
//! assert_eq!(
//!     call(Verb::Source, "urn:bayes:contact:stats", &[("as", "application/json")]),
//!     r#"{"cutoffs":{"ham":0.2,"spam":0.9},"messages":{"ham":1,"spam":1},"model":"contact","tokenizer":"1","tokens":10}"#
//! );
//! assert_eq!(call(Verb::Source, "urn:bayes:models", &[]), "contact");
//! assert_eq!(
//!     call(Verb::Source, "urn:bayes:models", &[("as", "application/json")]),
//!     r#"{"models":[{"name":"contact","readCapability":"urn:cap:bayes:read:contact","score":"urn:bayes:contact:score","stats":"urn:bayes:contact:stats","train":"urn:bayes:contact:train","trainCapability":"urn:cap:bayes:train:contact","untrain":"urn:bayes:contact:untrain"}]}"#
//! );
//! ```
#![forbid(unsafe_code)]

pub mod classify;
mod endpoints;
pub mod model;
pub mod store;
pub mod tokenize;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::{Arc, Mutex};

use ikigai_core::{ContentId, Error};

use crate::classify::Settings;
use crate::model::{Change, Label, Model, MAX_MODEL_BYTES};
use crate::store::{MemoryStore, Store};
use crate::tokenize::{tokenize, FIELDS};

pub use crate::endpoints::space;

/// The most models one [`Bayes`] offers.
pub const MAX_MODELS: usize = 64;

/// The longest model name, in characters.
pub const MAX_MODEL_NAME: usize = 40;

/// The largest message, in bytes: the body plus every field. Past it the message is refused,
/// never truncated.
pub const MAX_MESSAGE_BYTES: usize = 64 * 1024;

/// How many clues a score shows when the caller does not say (`clues=`).
pub const DEFAULT_CLUES_SHOWN: usize = 15;

/// A model name the host reserves: `urn:bayes:models` is the list of models.
const RESERVED: [&str; 1] = ["models"];

/// The read capability of a model: `urn:cap:bayes:read:{model}`.
pub fn read_cap(model: &str) -> String {
    format!("urn:cap:bayes:read:{model}")
}

/// The train capability of a model: `urn:cap:bayes:train:{model}`.
pub fn train_cap(model: &str) -> String {
    format!("urn:cap:bayes:train:{model}")
}

/// A model's resource IRI for `part` (`score`, `train`, `untrain`, `stats`).
pub fn model_iri(model: &str, part: &str) -> String {
    format!("urn:bayes:{model}:{part}")
}

/// Why a [`Bayes`] would not open.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenError(pub String);

impl fmt::Display for OpenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for OpenError {}

/// One model the host offers: its name and its settings.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    name: String,
    settings: Settings,
}

impl ModelConfig {
    /// A model with SpamBayes's default settings (ham below 0.20, spam at or above 0.90).
    pub fn new(name: impl Into<String>) -> Self {
        ModelConfig {
            name: name.into(),
            settings: Settings::default(),
        }
    }

    /// The same model with other cutoffs. Refused unless `0 <= ham <= spam <= 1`.
    pub fn cutoffs(mut self, ham: f64, spam: f64) -> Result<Self, OpenError> {
        self.settings = self
            .settings
            .with_cutoffs(ham, spam)
            .map_err(|e| OpenError(format!("model `{}`: {e}", self.name)))?;
        Ok(self)
    }

    /// The model's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The model's settings.
    pub fn settings(&self) -> &Settings {
        &self.settings
    }
}

/// Whether `name` is a model name: 1 to [`MAX_MODEL_NAME`] characters of lowercase ASCII letters,
/// digits and single hyphens, starting with a letter and not ending with a hyphen. The name is
/// spelled into IRIs, capabilities, file names and endpoint ids, so it is kept to the characters
/// all four carry unchanged.
///
/// ```
/// use ikigai_bayes::is_model_name;
///
/// assert!(is_model_name("contact"));
/// assert!(is_model_name("contact-2"));
/// assert!(!is_model_name("Contact"));
/// assert!(!is_model_name("a--b"));
/// assert!(!is_model_name("../etc"));
/// assert!(!is_model_name("models"));
/// ```
pub fn is_model_name(name: &str) -> bool {
    let len = name.chars().count();
    (1..=MAX_MODEL_NAME).contains(&len)
        && name.starts_with(|c: char| c.is_ascii_lowercase())
        && !name.ends_with('-')
        && !name.contains("--")
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        && !RESERVED.contains(&name)
}

/// A message: a body and, optionally, named fields from [`FIELDS`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    content: String,
    fields: Vec<(&'static str, String)>,
}

impl Message {
    /// A message with a body only.
    pub fn new(content: impl Into<String>) -> Self {
        Message {
            content: content.into(),
            fields: Vec::new(),
        }
    }

    /// The same message with a field set. A name outside [`FIELDS`] is refused; an empty or
    /// all-whitespace value is the field ABSENT (a form's blank optional field), so it changes
    /// neither the tokens nor the digest.
    pub fn field(mut self, name: &str, value: impl Into<String>) -> Result<Self, String> {
        let name = FIELDS
            .iter()
            .copied()
            .find(|f| *f == name)
            .ok_or_else(|| format!("`{name}` is not a field ({})", FIELDS.join(", ")))?;
        let value = value.into();
        self.fields.retain(|(f, _)| *f != name);
        if !value.trim().is_empty() {
            self.fields.push((name, value));
            self.fields
                .sort_by_key(|(f, _)| FIELDS.iter().position(|x| x == f));
        }
        Ok(self)
    }

    /// The body.
    pub fn content(&self) -> &str {
        &self.content
    }

    /// The size bounded by [`MAX_MESSAGE_BYTES`]: the body plus every field value.
    pub fn len(&self) -> usize {
        self.content.len() + self.fields.iter().map(|(_, v)| v.len()).sum::<usize>()
    }

    /// Whether the body and every field are empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The message's digest, which keys training. With no fields it is the BLAKE3 digest of the
    /// body's bytes, exactly what `b3sum` prints (as `b3:<hex>`). With fields it is the digest of
    /// a canonical encoding, so the same body with a different sender is a different message:
    ///
    /// ```text
    /// ikigai-bayes/message/1\n
    /// content\n<byte length>\n<body>\n
    /// <field>\n<byte length>\n<value>\n      (each present field, in FIELDS order)
    /// ```
    ///
    /// ```
    /// use ikigai_bayes::Message;
    /// use ikigai_core::ContentId;
    ///
    /// let plain = Message::new("hello");
    /// assert_eq!(plain.digest(), ContentId::of(b"hello"));
    /// let with_email = Message::new("hello").field("email", "a@example.com").unwrap();
    /// assert_ne!(with_email.digest(), plain.digest());
    /// // A blank field is no field.
    /// assert_eq!(Message::new("hello").field("organisation", " ").unwrap().digest(), plain.digest());
    /// ```
    pub fn digest(&self) -> ContentId {
        if self.fields.is_empty() {
            return ContentId::of(self.content.as_bytes());
        }
        let mut canonical = Vec::with_capacity(self.len() + 64);
        canonical.extend_from_slice(b"ikigai-bayes/message/1\n");
        let mut part = |name: &str, value: &str| {
            canonical.extend_from_slice(format!("{name}\n{}\n", value.len()).as_bytes());
            canonical.extend_from_slice(value.as_bytes());
            canonical.push(b'\n');
        };
        part("content", &self.content);
        for (name, value) in &self.fields {
            part(name, value);
        }
        ContentId::of(&canonical)
    }

    /// The message's token set ([`tokenize::tokenize`]).
    pub fn tokens(&self) -> Result<BTreeSet<String>, tokenize::TooManyTokens> {
        let fields: Vec<(&str, &str)> = self
            .fields
            .iter()
            .map(|(name, value)| (*name, value.as_str()))
            .collect();
        tokenize(&self.content, &fields)
    }
}

/// One model as the host holds it.
pub(crate) struct Slot {
    pub(crate) name: String,
    pub(crate) settings: Settings,
    pub(crate) model: Mutex<Model>,
}

/// The models a host offers, over the store that keeps them.
pub struct Bayes {
    slots: BTreeMap<String, Arc<Slot>>,
    store: Arc<dyn Store>,
}

impl fmt::Debug for Bayes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Bayes")
            .field("models", &self.slots.keys().collect::<Vec<_>>())
            .finish()
    }
}

/// What a train did, for the answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Trained {
    /// The message's digest.
    pub digest: ContentId,
    /// The label it now has.
    pub label: Label,
    /// What changed.
    pub change: Change,
    /// Spam and ham messages the model holds now.
    pub messages: (u64, u64),
}

/// What an untrain removed, for the answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Untrained {
    /// The message's digest.
    pub digest: ContentId,
    /// The label it had.
    pub label: Label,
    /// Spam and ham messages the model holds now.
    pub messages: (u64, u64),
}

impl Bayes {
    /// Open the models over a state directory the host chooses, taking its lock (one writer per
    /// directory). Each model is read from its file, or created empty if it is NEW; a model that
    /// is missing or corrupt is refused by name (see [`store`]).
    #[cfg(not(target_family = "wasm"))]
    pub fn open(
        state_dir: impl AsRef<std::path::Path>,
        models: impl IntoIterator<Item = ModelConfig>,
    ) -> Result<Bayes, OpenError> {
        let dir = state_dir.as_ref();
        let store = store::DirStore::open(dir)
            .map_err(|e| OpenError(format!("state directory {}: {e}", dir.display())))?;
        Bayes::with_store(Arc::new(store), models)
    }

    /// Open the models in memory: nothing survives the process.
    pub fn in_memory(models: impl IntoIterator<Item = ModelConfig>) -> Result<Bayes, OpenError> {
        Bayes::with_store(Arc::new(MemoryStore::new()), models)
    }

    /// Open the models over any [`Store`].
    pub fn with_store(
        store: Arc<dyn Store>,
        models: impl IntoIterator<Item = ModelConfig>,
    ) -> Result<Bayes, OpenError> {
        let mut slots = BTreeMap::new();
        for config in models {
            let name = config.name;
            if !is_model_name(&name) {
                return Err(OpenError(format!(
                    "`{name}` is not a model name: 1 to {MAX_MODEL_NAME} lowercase letters, \
                     digits and single hyphens, starting with a letter (and not `models`)"
                )));
            }
            if slots.contains_key(&name) {
                return Err(OpenError(format!("model `{name}` is offered twice")));
            }
            if slots.len() == MAX_MODELS {
                return Err(OpenError(format!(
                    "more than {MAX_MODELS} models offered (refused at `{name}`)"
                )));
            }
            let model = match store
                .load(&name)
                .map_err(|e| OpenError(format!("model `{name}`: {e}")))?
            {
                Some(bytes) => Model::from_bytes(&name, &bytes)
                    .map_err(|e| OpenError(format!("model `{name}`: its file is refused: {e}")))?,
                None => {
                    // New: created now, so that from here on it exists and a missing file is
                    // a loss the store can name.
                    let model = Model::new(&name);
                    store
                        .save(&name, &model.to_bytes())
                        .map_err(|e| OpenError(format!("model `{name}`: create: {e}")))?;
                    model
                }
            };
            slots.insert(
                name.clone(),
                Arc::new(Slot {
                    name,
                    settings: config.settings,
                    model: Mutex::new(model),
                }),
            );
        }
        Ok(Bayes { slots, store })
    }

    /// The models offered, by name.
    pub fn models(&self) -> impl Iterator<Item = &str> {
        self.slots.keys().map(String::as_str)
    }

    pub(crate) fn slots(&self) -> impl Iterator<Item = &Arc<Slot>> {
        self.slots.values()
    }

    pub(crate) fn store(&self) -> &Arc<dyn Store> {
        &self.store
    }
}

/// Teach `slot`'s model a message, persisting it before the answer. Nothing changes (in memory or
/// on disk) when the train is refused or the write fails.
pub(crate) fn train(
    store: &dyn Store,
    slot: &Slot,
    message: &Message,
    label: Label,
) -> Result<Trained, Error> {
    let tokens = message.tokens().map_err(too_many)?;
    let digest = message.digest();
    let mut model = slot.model.lock().expect("model lock");
    let change = model
        .train(digest, label, &tokens)
        .map_err(|r| Error::Conflict(format!("model `{}`: {r}", slot.name)))?;
    if change != Change::Unchanged {
        if let Err(e) = persist(store, &slot.name, &model) {
            model.revert(digest, change);
            return Err(e);
        }
    }
    Ok(Trained {
        digest,
        label,
        change,
        messages: (model.nspam(), model.nham()),
    })
}

/// Remove a message from `slot`'s model, persisting it before the answer.
pub(crate) fn untrain(
    store: &dyn Store,
    slot: &Slot,
    digest: ContentId,
) -> Result<Untrained, Error> {
    let mut model = slot.model.lock().expect("model lock");
    let tokens = model
        .tokens_of(&digest)
        .map(<[String]>::to_vec)
        .ok_or_else(|| {
            Error::NotFound(format!(
                "model `{}` was never taught message {digest}",
                slot.name
            ))
        })?;
    let label = model.untrain(&digest).expect("the message is trained");
    if let Err(e) = persist(store, &slot.name, &model) {
        model.restore(digest, label, tokens);
        return Err(e);
    }
    Ok(Untrained {
        digest,
        label,
        messages: (model.nspam(), model.nham()),
    })
}

fn persist(store: &dyn Store, name: &str, model: &Model) -> Result<(), Error> {
    let bytes = model.to_bytes();
    if bytes.len() > MAX_MODEL_BYTES {
        return Err(Error::Conflict(format!(
            "model `{name}` would be {} bytes, past the bound of {MAX_MODEL_BYTES}: untrain old \
             messages or start another model",
            bytes.len()
        )));
    }
    store
        .save(name, &bytes)
        .map_err(|e| Error::Endpoint(format!("model `{name}`: {e}")))
}

pub(crate) fn too_many(e: tokenize::TooManyTokens) -> Error {
    Error::InvalidArgument {
        name: "content".into(),
        detail: e.to_string(),
    }
}
