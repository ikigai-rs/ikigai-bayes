//! A model: the set of messages it was taught, and the token counts derived from them.
//!
//! The PERSISTED state is the trained messages, each as its content digest, its label and its
//! token set. The counts (per token, how many spam and ham messages hold it; how many messages of
//! each label) are an index DERIVED from that set, rebuilt when a model is read. So a count can
//! never drift from the messages it counts, and untraining a message leaves the model exactly as
//! if it had never been taught it, by construction: the counts are a sum over messages, and the
//! message is gone from the sum.
//!
//! That is also why the distinct-token bound REFUSES rather than prunes: pruning a rare token
//! would make the counts something other than that sum, and then neither exact untraining nor a
//! model rebuilt from its file would hold.
//!
//! ⚠ The state is as sensitive as a bag of words of every trained message: the token set is
//! kept per message so that untraining by digest alone stays exact. Keep the state directory as
//! private as the messages.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::sync::OnceLock;

use ikigai_core::ContentId;
use serde::{Deserialize, Serialize};

use crate::classify::{fisher, robinson, Class, Settings};
use crate::tokenize::TOKENIZER_VERSION;

/// The most messages one model may hold. Past it, training a NEW message is refused (a move or
/// a repeat is not new).
pub const MAX_MESSAGES: usize = 50_000;

/// The most distinct tokens one model may hold. Past it, training a message that brings new
/// tokens is refused: untrain old messages or start another model. A refusal, not pruning, so
/// that the counts stay an exact sum over the trained messages (see the module documentation).
pub const MAX_DISTINCT_TOKENS: usize = 250_000;

/// The largest model file read or written, in bytes. A train whose model would serialize past it
/// is refused; a file past it is refused at open.
pub const MAX_MODEL_BYTES: usize = 64 * 1024 * 1024;

/// The longest token a model file may hold, in bytes. The tokenizer never produces one near it
/// (a word is at most [`MAX_WORD`](crate::tokenize::MAX_WORD) characters and a host at most 253
/// bytes); a longer one marks a file this crate did not write.
pub const MAX_TOKEN_BYTES: usize = 512;

/// The `format` a model file declares.
pub const FORMAT: &str = "ikigai-bayes/model";

/// The format version this crate reads and writes.
pub const FORMAT_VERSION: u32 = 1;

/// A training label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Label {
    /// Unwanted.
    Spam,
    /// Wanted.
    Ham,
}

impl Label {
    /// `spam` or `ham`.
    pub fn as_str(self) -> &'static str {
        match self {
            Label::Spam => "spam",
            Label::Ham => "ham",
        }
    }

    /// The other label.
    pub fn other(self) -> Label {
        match self {
            Label::Spam => Label::Ham,
            Label::Ham => Label::Spam,
        }
    }

    /// Parse `spam` or `ham`.
    pub fn parse(s: &str) -> Option<Label> {
        match s {
            "spam" => Some(Label::Spam),
            "ham" => Some(Label::Ham),
            _ => None,
        }
    }
}

/// The names for the high-score and low-score sides, respectively.
/// Internal `Label::Spam`/`Label::Ham` remain stable storage slots.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "LabelNames")]
pub struct Labels {
    high: String,
    low: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LabelNames {
    high: String,
    low: String,
}

impl TryFrom<LabelNames> for Labels {
    type Error = String;

    fn try_from(names: LabelNames) -> Result<Self, Self::Error> {
        Self::new(names.high, names.low)
    }
}

impl Default for Labels {
    fn default() -> Self {
        Self {
            high: "spam".into(),
            low: "ham".into(),
        }
    }
}

impl Labels {
    /// Two distinct names, using lowercase letters, digits and single hyphens.
    /// `unsure` is reserved for the band between the cutoffs.
    pub fn new(high: impl Into<String>, low: impl Into<String>) -> Result<Self, String> {
        let labels = Self {
            high: high.into(),
            low: low.into(),
        };
        if [&labels.high, &labels.low]
            .iter()
            .any(|s| !crate::is_model_name(s) || s.as_str() == "unsure")
            || labels.high == labels.low
        {
            return Err("labels must be distinct lowercase names (1 to 40 characters); `unsure` and `models` are reserved".into());
        }
        Ok(labels)
    }

    /// The public name for an internal training slot.
    pub fn name(&self, label: Label) -> &str {
        match label {
            Label::Spam => &self.high,
            Label::Ham => &self.low,
        }
    }

    /// Parse one of this model's two names.
    pub fn parse(&self, text: &str) -> Option<Label> {
        if text == self.high {
            Some(Label::Spam)
        } else if text == self.low {
            Some(Label::Ham)
        } else {
            None
        }
    }

    /// The public class name, including the reserved `unsure` class.
    pub fn class(&self, class: Class) -> &str {
        match class {
            Class::Spam => &self.high,
            Class::Ham => &self.low,
            Class::Unsure => "unsure",
        }
    }

    fn is_default(&self) -> bool {
        self == &Self::default()
    }
}

/// What a train did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Change {
    /// The message was new to the model.
    Added,
    /// The message was trained with the other label and now has this one.
    Moved {
        /// The label it had.
        from: Label,
    },
    /// The message was already trained with this label: nothing changed.
    Unchanged,
}

/// Why a train was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// The model holds [`MAX_MESSAGES`] messages.
    TooManyMessages,
    /// The message would take the model past [`MAX_DISTINCT_TOKENS`].
    TooManyTokens {
        /// Distinct tokens the model holds now.
        held: usize,
        /// Tokens the message would add.
        new: usize,
    },
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refusal::TooManyMessages => write!(
                f,
                "the model holds {MAX_MESSAGES} messages, its bound: untrain old messages or \
                 start another model"
            ),
            Refusal::TooManyTokens { held, new } => write!(
                f,
                "the model holds {held} distinct tokens and this message adds {new}, past the \
                 bound of {MAX_DISTINCT_TOKENS}: untrain old messages or start another model \
                 (refused, not pruned, so the counts stay exact)"
            ),
        }
    }
}

/// One trained message.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Trained {
    label: Label,
    tokens: Vec<String>,
}

/// A model's counts for one token: spam messages, ham messages.
type Counts = (u64, u64);

/// A model.
#[derive(Debug, Clone)]
pub struct Model {
    name: String,
    labels: Labels,
    messages: BTreeMap<ContentId, Trained>,
    counts: HashMap<String, Counts>,
    nspam: u64,
    nham: u64,
    digest: OnceLock<ContentId>,
}

impl PartialEq for Model {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.labels == other.labels
            && self.messages == other.messages
            && self.counts == other.counts
            && self.nspam == other.nspam
            && self.nham == other.nham
    }
}
impl Eq for Model {}

/// One clue of a score: a token and its probability.
#[derive(Debug, Clone, PartialEq)]
pub struct Clue {
    /// The token.
    pub token: String,
    /// Robinson's f(w) for it.
    pub probability: f64,
}

/// A score.
#[derive(Debug, Clone, PartialEq)]
pub struct Score {
    /// Fisher's combined score, in `[0, 1]`.
    pub score: f64,
    /// Digest of the trained state used for this score.
    pub model_digest: ContentId,
    /// The class under the model's cutoffs.
    pub class: Class,
    /// Every clue combined, strongest first (furthest from 0.5; ties by token).
    pub clues: Vec<Clue>,
}

/// The persisted shape.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    format: String,
    version: u32,
    model: String,
    tokenizer: String,
    #[serde(default, skip_serializing_if = "Labels::is_default")]
    labels: Labels,
    messages: Vec<FileMessage>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileMessage {
    digest: String,
    label: Label,
    tokens: Vec<String>,
}

impl Model {
    /// An empty model.
    pub fn new(name: impl Into<String>) -> Self {
        Model {
            name: name.into(),
            labels: Labels::default(),
            messages: BTreeMap::new(),
            counts: HashMap::new(),
            nspam: 0,
            nham: 0,
            digest: OnceLock::new(),
        }
    }

    /// An empty model with declared high-score and low-score labels.
    pub fn with_labels(name: impl Into<String>, labels: Labels) -> Self {
        Self {
            labels,
            ..Self::new(name)
        }
    }

    /// The model's declared labels.
    pub fn labels(&self) -> &Labels {
        &self.labels
    }

    /// A deterministic BLAKE3 digest of the trained state, excluding the model name.
    /// No-op training and reads leave it unchanged; exact untraining restores prior digests.
    /// The encoding is the model file with an empty model name, including tokenizer and labels.
    pub fn digest(&self) -> ContentId {
        *self
            .digest
            .get_or_init(|| ContentId::of(&self.file_bytes("")))
    }

    /// The model's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Spam messages trained.
    pub fn nspam(&self) -> u64 {
        self.nspam
    }

    /// Ham messages trained.
    pub fn nham(&self) -> u64 {
        self.nham
    }

    /// Distinct tokens held.
    pub fn distinct_tokens(&self) -> usize {
        self.counts.len()
    }

    /// The label a message was trained with, if it was.
    pub fn label_of(&self, digest: &ContentId) -> Option<Label> {
        self.messages.get(digest).map(|t| t.label)
    }

    /// A token's counts: (spam messages holding it, ham messages holding it).
    pub fn counts(&self, token: &str) -> (u64, u64) {
        self.counts.get(token).copied().unwrap_or((0, 0))
    }

    /// Teach the model a message. Idempotent per digest: the same message with the same label is
    /// [`Change::Unchanged`]; with the other label it MOVES ([`Change::Moved`]), which is how a
    /// correction works.
    pub fn train(
        &mut self,
        digest: ContentId,
        label: Label,
        tokens: &BTreeSet<String>,
    ) -> Result<Change, Refusal> {
        if let Some(previous) = self.messages.get(&digest).map(|t| t.label) {
            if previous == label {
                return Ok(Change::Unchanged);
            }
            // A move: the same message under the other label. Its tokens are the ones it was
            // trained with, so the move adds no token and needs no bound.
            self.digest.take();
            let trained = self.messages.remove(&digest).expect("present");
            self.add_counts(&trained.tokens, previous, -1);
            self.add_counts(&trained.tokens, label, 1);
            self.messages.insert(
                digest,
                Trained {
                    label,
                    tokens: trained.tokens,
                },
            );
            return Ok(Change::Moved { from: previous });
        }
        if self.messages.len() >= MAX_MESSAGES {
            return Err(Refusal::TooManyMessages);
        }
        let new = tokens
            .iter()
            .filter(|t| !self.counts.contains_key(*t))
            .count();
        if self.counts.len() + new > MAX_DISTINCT_TOKENS {
            return Err(Refusal::TooManyTokens {
                held: self.counts.len(),
                new,
            });
        }
        self.digest.take();
        let tokens: Vec<String> = tokens.iter().cloned().collect();
        self.add_counts(&tokens, label, 1);
        self.messages.insert(digest, Trained { label, tokens });
        Ok(Change::Added)
    }

    /// Remove a trained message, leaving the model exactly as if it had never been taught it.
    /// The label it had, or `None` when it was not trained.
    pub fn untrain(&mut self, digest: &ContentId) -> Option<Label> {
        let trained = self.messages.remove(digest)?;
        self.digest.take();
        self.add_counts(&trained.tokens, trained.label, -1);
        Some(trained.label)
    }

    /// Undo a [`train`](Self::train) that returned `change` (used when persisting it failed).
    pub(crate) fn revert(&mut self, digest: ContentId, change: Change) {
        match change {
            Change::Added => {
                self.untrain(&digest);
            }
            Change::Moved { from } => {
                // A move back: the message is present under the other label, so its stored
                // tokens are used and the empty set passed here is never read.
                let moved = self.train(digest, from, &BTreeSet::new());
                debug_assert_eq!(moved, Ok(Change::Moved { from: from.other() }));
            }
            Change::Unchanged => {}
        }
    }

    /// Re-teach a message removed by [`untrain`](Self::untrain) (used when persisting it failed).
    pub(crate) fn restore(&mut self, digest: ContentId, label: Label, tokens: Vec<String>) {
        self.digest.take();
        self.add_counts(&tokens, label, 1);
        self.messages.insert(digest, Trained { label, tokens });
    }

    /// The token set a trained message was taught with.
    pub(crate) fn tokens_of(&self, digest: &ContentId) -> Option<&[String]> {
        self.messages.get(digest).map(|t| t.tokens.as_slice())
    }

    fn add_counts(&mut self, tokens: &[String], label: Label, delta: i64) {
        let bump = |n: &mut u64| {
            *n = if delta > 0 {
                *n + 1
            } else {
                n.checked_sub(1).expect("a count never goes below zero")
            }
        };
        match label {
            Label::Spam => bump(&mut self.nspam),
            Label::Ham => bump(&mut self.nham),
        }
        for token in tokens {
            let entry = self.counts.entry(token.clone()).or_insert((0, 0));
            match label {
                Label::Spam => bump(&mut entry.0),
                Label::Ham => bump(&mut entry.1),
            }
            if *entry == (0, 0) {
                self.counts.remove(token);
            }
        }
    }

    /// Score a message's token set.
    pub fn score(&self, tokens: &BTreeSet<String>, settings: &Settings) -> Score {
        let mut clues: Vec<Clue> = tokens
            .iter()
            .filter_map(|token| {
                let (s, h) = self.counts(token);
                let p = robinson(s, h, self.nspam, self.nham, settings);
                ((p - 0.5).abs() >= settings.minimum_prob_strength).then(|| Clue {
                    token: token.clone(),
                    probability: p,
                })
            })
            .collect();
        clues.sort_by(|a, b| {
            let da = (a.probability - 0.5).abs();
            let db = (b.probability - 0.5).abs();
            db.total_cmp(&da).then_with(|| a.token.cmp(&b.token))
        });
        clues.truncate(settings.max_discriminators);
        let probabilities: Vec<f64> = clues.iter().map(|c| c.probability).collect();
        let score = fisher(&probabilities);
        Score {
            score,
            model_digest: self.digest(),
            class: settings.class_of(score),
            clues,
        }
    }

    /// The model file's bytes: deterministic (messages sorted by digest, tokens sorted).
    pub fn to_bytes(&self) -> Vec<u8> {
        self.file_bytes(&self.name)
    }

    fn file_bytes(&self, name: &str) -> Vec<u8> {
        let file = File {
            format: FORMAT.to_string(),
            version: FORMAT_VERSION,
            model: name.to_owned(),
            tokenizer: TOKENIZER_VERSION.to_string(),
            labels: self.labels.clone(),
            messages: self
                .messages
                .iter()
                .map(|(digest, t)| FileMessage {
                    digest: digest.to_string(),
                    label: t.label,
                    tokens: t.tokens.clone(),
                })
                .collect(),
        };
        serde_json::to_vec(&file).expect("a model always serializes")
    }

    /// Read a model file, refusing anything this crate would not have written: another format or
    /// version, another model's name, another tokenizer version, a malformed or repeated digest,
    /// an unsorted or oversized token set, or a file past any bound.
    pub fn from_bytes(name: &str, bytes: &[u8]) -> Result<Model, String> {
        if bytes.len() > MAX_MODEL_BYTES {
            return Err(format!(
                "{} bytes, past the bound of {MAX_MODEL_BYTES}",
                bytes.len()
            ));
        }
        let file: File = serde_json::from_slice(bytes).map_err(|e| format!("not a model: {e}"))?;
        if file.format != FORMAT || file.version != FORMAT_VERSION {
            return Err(format!(
                "format `{}` version {}, not `{FORMAT}` version {FORMAT_VERSION}",
                file.format, file.version
            ));
        }
        if file.model != name {
            return Err(format!(
                "it is model `{}`'s file, not `{name}`'s",
                file.model
            ));
        }
        if file.tokenizer != TOKENIZER_VERSION {
            return Err(format!(
                "trained with tokenizer version {}, and this build tokenizes with version \
                 {TOKENIZER_VERSION}: the same text would count under different tokens, so the \
                 model must be retrained",
                file.tokenizer
            ));
        }
        if file.messages.len() > MAX_MESSAGES {
            return Err(format!(
                "{} messages, past the bound of {MAX_MESSAGES}",
                file.messages.len()
            ));
        }
        let mut model = Model::with_labels(name, file.labels);
        for m in file.messages {
            let digest = ContentId::parse(&m.digest)
                .map_err(|e| format!("message digest `{}`: {e}", m.digest))?;
            if model.messages.contains_key(&digest) {
                return Err(format!("message {digest} appears twice"));
            }
            if m.tokens.len() > crate::tokenize::MAX_TOKENS {
                return Err(format!("message {digest} holds {} tokens", m.tokens.len()));
            }
            if m.tokens.windows(2).any(|w| w[0] >= w[1]) {
                return Err(format!(
                    "message {digest}'s tokens are not sorted and distinct"
                ));
            }
            if let Some(t) = m.tokens.iter().find(|t| t.len() > MAX_TOKEN_BYTES) {
                return Err(format!(
                    "message {digest} holds a token of {} bytes",
                    t.len()
                ));
            }
            model.add_counts(&m.tokens, m.label, 1);
            model.messages.insert(
                digest,
                Trained {
                    label: m.label,
                    tokens: m.tokens,
                },
            );
        }
        if model.counts.len() > MAX_DISTINCT_TOKENS {
            return Err(format!(
                "{} distinct tokens, past the bound of {MAX_DISTINCT_TOKENS}",
                model.counts.len()
            ));
        }
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenize::tokenize;

    fn d(s: &str) -> ContentId {
        ContentId::of(s.as_bytes())
    }

    fn t(s: &str) -> BTreeSet<String> {
        tokenize(s, &[]).unwrap()
    }

    #[test]
    fn training_twice_counts_once() {
        let mut m = Model::new("m");
        assert_eq!(
            m.train(d("a"), Label::Spam, &t("cheap pills")),
            Ok(Change::Added)
        );
        let once = m.clone();
        assert_eq!(
            m.train(d("a"), Label::Spam, &t("cheap pills")),
            Ok(Change::Unchanged)
        );
        assert_eq!(m, once);
        assert_eq!(m.nspam(), 1);
        assert_eq!(m.counts("cheap"), (1, 0));
    }

    #[test]
    fn training_with_the_other_label_moves() {
        let mut m = Model::new("m");
        m.train(d("a"), Label::Spam, &t("cheap pills")).unwrap();
        assert_eq!(
            m.train(d("a"), Label::Ham, &t("cheap pills")),
            Ok(Change::Moved { from: Label::Spam })
        );
        assert_eq!((m.nspam(), m.nham()), (0, 1));
        assert_eq!(m.counts("cheap"), (0, 1));
        // Equal to a model only ever taught it as ham.
        let mut only_ham = Model::new("m");
        only_ham
            .train(d("a"), Label::Ham, &t("cheap pills"))
            .unwrap();
        assert_eq!(m, only_ham);
    }

    #[test]
    fn untraining_restores_the_exact_prior_state() {
        let mut m = Model::new("m");
        m.train(d("x"), Label::Ham, &t("meeting on tuesday"))
            .unwrap();
        m.train(d("y"), Label::Spam, &t("cheap pills tuesday"))
            .unwrap();
        let before = m.clone();
        let bytes = m.to_bytes();
        m.train(d("z"), Label::Spam, &t("cheap watches now"))
            .unwrap();
        assert_ne!(m, before);
        assert_eq!(m.untrain(&d("z")), Some(Label::Spam));
        assert_eq!(m, before);
        assert_eq!(m.to_bytes(), bytes, "byte-identical, not just equal counts");
        assert_eq!(m.untrain(&d("z")), None);
    }

    #[test]
    fn a_count_that_reaches_zero_is_gone() {
        let mut m = Model::new("m");
        m.train(d("x"), Label::Spam, &t("unique words here"))
            .unwrap();
        m.untrain(&d("x"));
        assert_eq!(m.distinct_tokens(), 0);
        assert_eq!(m, Model::new("m"));
    }

    #[test]
    fn the_file_round_trips_and_rebuilds_the_counts() {
        let mut m = Model::new("contact");
        m.train(d("x"), Label::Ham, &t("meeting on tuesday"))
            .unwrap();
        m.train(d("y"), Label::Spam, &t("cheap pills")).unwrap();
        let back = Model::from_bytes("contact", &m.to_bytes()).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn a_file_this_crate_did_not_write_is_refused() {
        let mut m = Model::new("contact");
        m.train(d("x"), Label::Ham, &t("meeting on tuesday"))
            .unwrap();
        let good = String::from_utf8(m.to_bytes()).unwrap();
        let refused = |bytes: &str| Model::from_bytes("contact", bytes.as_bytes()).unwrap_err();
        assert!(refused("{").contains("not a model"));
        assert!(Model::from_bytes("other", good.as_bytes())
            .unwrap_err()
            .contains("model `contact`'s file"));
        assert!(
            refused(&good.replace("\"tokenizer\":\"1\"", "\"tokenizer\":\"0\""))
                .contains("retrained")
        );
        assert!(refused(&good.replace("\"version\":1", "\"version\":2")).contains("format"));
        assert!(refused(&good.replace("\"digest\":\"b3:", "\"digest\":\"md5:")).contains("digest"));
        assert!(
            refused(&good.replace("\"messages\"", "\"extra\":1,\"messages\""))
                .contains("not a model")
        );
    }

    #[test]
    fn unsorted_tokens_are_refused() {
        let file = format!(
            r#"{{"format":"{FORMAT}","version":1,"model":"m","tokenizer":"1","messages":[{{"digest":"{}","label":"spam","tokens":["b","a"]}}]}}"#,
            d("x")
        );
        assert!(Model::from_bytes("m", file.as_bytes())
            .unwrap_err()
            .contains("not sorted"));
    }

    #[test]
    fn a_repeated_digest_is_refused() {
        let msg = format!(r#"{{"digest":"{}","label":"spam","tokens":["a"]}}"#, d("x"));
        let file = format!(
            r#"{{"format":"{FORMAT}","version":1,"model":"m","tokenizer":"1","messages":[{msg},{msg}]}}"#
        );
        assert!(Model::from_bytes("m", file.as_bytes())
            .unwrap_err()
            .contains("twice"));
    }

    #[test]
    fn an_untrained_model_scores_every_message_unsure() {
        let m = Model::new("m");
        let s = m.score(&t("anything at all"), &Settings::default());
        assert_eq!(s.score, 0.5);
        assert_eq!(s.class, Class::Unsure);
        assert!(s.clues.is_empty());
    }

    #[test]
    fn clues_are_strongest_first_and_bounded() {
        let mut m = Model::new("m");
        m.train(d("s1"), Label::Spam, &t("cheap pills cheap"))
            .unwrap();
        m.train(d("s2"), Label::Spam, &t("cheap watches")).unwrap();
        m.train(d("h1"), Label::Ham, &t("meeting watches")).unwrap();
        let settings = Settings::default();
        let s = m.score(&t("cheap watches meeting unknownword"), &settings);
        let tokens: Vec<&str> = s.clues.iter().map(|c| c.token.as_str()).collect();
        // `cheap`: 2 of 2 spam, 0 of 1 ham, so f = 2.225/2.45 ≈ 0.908 (0.408 from 0.5).
        // `meeting`: 0 of 2 spam, 1 of 1 ham, so f = 0.225/1.45 ≈ 0.155 (0.345 from 0.5).
        // `watches`: 1 of 2 spam, 1 of 1 ham, p = 1/3, f ≈ 0.364 (0.136 from 0.5).
        // The coarse tokens are in every message, so their ratios match and f = 0.5: no clue.
        assert_eq!(tokens[0], "cheap", "{tokens:?}");
        for w in s.clues.windows(2) {
            assert!((w[0].probability - 0.5).abs() >= (w[1].probability - 0.5).abs());
        }
        assert!(!tokens.contains(&"unknownword"));
        let one = Settings {
            max_discriminators: 1,
            ..settings
        };
        assert_eq!(m.score(&t("cheap watches meeting"), &one).clues.len(), 1);
    }

    #[test]
    fn the_message_bound_refuses_new_messages_only() {
        let mut m = Model::new("m");
        for i in 0..MAX_MESSAGES {
            m.messages.insert(
                d(&i.to_string()),
                Trained {
                    label: Label::Ham,
                    tokens: Vec::new(),
                },
            );
        }
        // The counts those messages would have built (they hold no tokens).
        m.nham = MAX_MESSAGES as u64;
        assert_eq!(
            m.train(d("new"), Label::Spam, &t("hello")),
            Err(Refusal::TooManyMessages)
        );
        // A message already trained may still move.
        assert_eq!(
            m.train(d("0"), Label::Spam, &BTreeSet::new()),
            Ok(Change::Moved { from: Label::Ham })
        );
    }

    #[test]
    fn the_token_bound_refuses_rather_than_prunes() {
        let mut m = Model::new("m");
        for i in 0..MAX_DISTINCT_TOKENS - 1 {
            m.counts.insert(format!("tok{i}"), (1, 0));
        }
        let before = m.clone();
        // Two new tokens (plus coarse ones): past the bound by at least one.
        let err = m
            .train(d("x"), Label::Spam, &t("brandnew another"))
            .unwrap_err();
        assert!(matches!(err, Refusal::TooManyTokens { .. }), "{err:?}");
        assert!(err.to_string().contains("refused, not pruned"));
        assert_eq!(m, before, "a refused train changes nothing");
    }
}
