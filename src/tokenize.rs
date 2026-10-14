//! The tokenizer: a message (a body and, optionally, named fields) to the SET of tokens a model
//! counts.
//!
//! A tokenizer change invalidates every model trained with the old one (the same text would
//! count under different tokens), so the tokenizer is VERSIONED: [`TOKENIZER_VERSION`] is
//! recorded in every model file, and a model trained with another version is refused at open
//! rather than scored with tokens it never saw. Every rule below is pinned by a test, and
//! changing any of them is a version bump.
//!
//! # The rules (version `1`)
//!
//! The body is split on whitespace into raw words. Each raw word is, in this order:
//!
//! - **A URL** when it begins (case-insensitively) with `http://`, `https://` or `www.`. It is
//!   reduced to three tokens, and its path and query are dropped (they are mostly per-message
//!   noise): `url-scheme:<scheme>` (`none` for a bare `www.`), `url-host:<host>` and
//!   `url-domain:<registrable domain>`. A host that is an IPv4 literal is `url-host:ip-literal`
//!   and `url-domain:ip-literal` (a literal address is itself the signal). A host over 253
//!   bytes, or one holding a character a host cannot, is `url-host:malformed`. Each URL counts
//!   as one link.
//! - **An email address** when it holds an `@` with a non-empty local part and a dotted domain
//!   of letters, digits, `-` and `.`: one token, `addr:<domain>`. The local part is dropped: it
//!   identifies a person, and the domain is what carries across messages.
//! - **A word** otherwise. Leading and trailing ASCII punctuation is trimmed and the rest is
//!   lowercased (Unicode lowercasing). A word of fewer than [`MIN_WORD`] characters is dropped
//!   (SpamBayes's rule: short words are mostly noise). A word of more than [`MAX_WORD`]
//!   characters becomes `skip:<first character> <length rounded down to 10>`, SpamBayes's
//!   "skip" token, so a run of base64 or an unbroken CJK sentence still says something
//!   without minting a token per blob.
//!
//! Then three coarse tokens describe the body as a whole, each a bucket rather than a count so
//! that they generalize:
//!
//! - `links:<n>` — how many URLs: `0`, `1`, `2`, `3-5`, `6+`.
//! - `caps-runs:<n>` — how many runs of consecutive ALL-CAPS words, a word being all-caps when
//!   it has at least three letters and every letter is uppercase: `0`, `1`, `2-3`, `4+`.
//! - `non-ascii:<p>` — the share of non-whitespace characters that are not ASCII, in percent:
//!   `0`, `1-9`, `10-29`, `30-59`, `60-100`.
//!
//! **Fields** ([`FIELDS`]: a contact form's name, email and organization, the last under the
//! form's existing identifier `organisation`; an email's subject and from) are tokenized by the
//! same word, URL and address rules, and every token is prefixed with
//! the field's name and a colon (`subject:viagra`, `name:url-domain:example.com`), so the same
//! word means different things in different places. The two ADDRESS fields (`from`, `email`)
//! are special: a value that is one address becomes `<field>:<domain>` (`from:example.com`),
//! and a value that is not one becomes `<field>:no-address` plus its prefixed words. Fields get
//! no coarse tokens.
//!
//! A model counts each token at most once per message (SpamBayes's rule): the answer is a SET.
//!
//! # What it does not do
//!
//! No HTML parsing, no MIME decoding, no header mining: the input is the plain text a contact
//! form or a caller hands over. The registrable domain is an APPROXIMATION, not the Public
//! Suffix List: the last two labels, or the last three when the top-level label has two letters
//! and the one before it is a common second-level label (`co`, `com`, `org`, `ac`, …), so
//! `shop.example.co.uk` is `example.co.uk`. A list would be a data dependency whose every update
//! changes the tokenizer.
//!
//! # Bounds
//!
//! Tokenizing is a single linear pass with no recursion (there is nothing nested to recurse
//! into). The number of tokens PRODUCED (before the set removes duplicates) is bounded by
//! [`MAX_TOKENS`]; past it the message is refused, never truncated, because a truncated message
//! would be scored and trained on a prefix the caller did not choose.
//!
//! ```
//! use ikigai_bayes::tokenize::tokenize;
//!
//! let tokens = tokenize("CHEAP pills at https://www.shop.example.co.uk/buy now", &[]).unwrap();
//! let tokens: Vec<&str> = tokens.iter().map(String::as_str).collect();
//! assert_eq!(
//!     tokens,
//!     [
//!         "caps-runs:1",
//!         "cheap",
//!         "links:1",
//!         "non-ascii:0",
//!         "now",
//!         "pills",
//!         "url-domain:example.co.uk",
//!         "url-host:www.shop.example.co.uk",
//!         "url-scheme:https",
//!     ]
//! );
//! ```

use std::collections::BTreeSet;
use std::fmt;

/// The tokenizer's version, recorded in every model it trains. Bumped by any change to the
/// rules in this module's documentation.
pub const TOKENIZER_VERSION: &str = "1";

/// The shortest word kept, in characters (SpamBayes: 3).
pub const MIN_WORD: usize = 3;

/// The longest word kept as itself, in characters; a longer one becomes a `skip:` token.
/// SpamBayes uses 12; 20 keeps ordinary long English words (`internationalization` is 20) as
/// themselves.
pub const MAX_WORD: usize = 20;

/// The most tokens one message may PRODUCE, before duplicates are removed. Past it the message
/// is refused. With [`MAX_MESSAGE_BYTES`](crate::MAX_MESSAGE_BYTES) at 64 KiB, a message of
/// three-letter words separated by single spaces produces at most 16,384 words, so this bound is
/// reached only by input that is mostly tokens with nothing between them.
pub const MAX_TOKENS: usize = 16_384;

/// The longest host kept as a token, in bytes: the DNS limit.
const MAX_HOST: usize = 253;

/// The named fields a caller may pass beside the body, each tokenized under its own prefix.
/// `organisation` keeps the contact form's existing field identifier.
pub const FIELDS: [&str; 5] = ["subject", "from", "name", "email", "organisation"];

/// The fields whose value is expected to be one email address.
pub const ADDRESS_FIELDS: [&str; 2] = ["from", "email"];

/// Second-level labels under a two-letter country code that a registrable domain reaches past.
const SECOND_LEVEL: [&str; 12] = [
    "ac", "co", "com", "edu", "gov", "ltd", "net", "ne", "or", "org", "plc", "gen",
];

/// A message that produced more than [`MAX_TOKENS`] tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TooManyTokens {
    /// How many tokens had been produced when the bound was crossed.
    pub produced: usize,
}

impl fmt::Display for TooManyTokens {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the message produces more than {MAX_TOKENS} tokens (refused, not truncated: a \
             truncated message would be judged on a prefix nobody chose)"
        )
    }
}

impl std::error::Error for TooManyTokens {}

/// Tokenize a body and its named fields into the set a model counts. `fields` are
/// `(name, value)` pairs; a name outside [`FIELDS`] is the caller's mistake and is not checked
/// here (the endpoints accept only declared fields).
pub fn tokenize(body: &str, fields: &[(&str, &str)]) -> Result<BTreeSet<String>, TooManyTokens> {
    let mut sink = Sink::default();
    let shape = words(body, "", &mut sink)?;
    sink.push(format!("links:{}", links_bucket(shape.links)))?;
    sink.push(format!("caps-runs:{}", caps_bucket(shape.caps_runs)))?;
    sink.push(format!(
        "non-ascii:{}",
        non_ascii_bucket(shape.non_ascii, shape.non_space)
    ))?;
    for (name, value) in fields {
        let prefix = format!("{name}:");
        if ADDRESS_FIELDS.contains(name) {
            match address_domain(value.trim()) {
                Some(domain) => {
                    sink.push(format!("{prefix}{domain}"))?;
                    continue;
                }
                None => sink.push(format!("{prefix}no-address"))?,
            }
        }
        words(value, &prefix, &mut sink)?;
    }
    Ok(sink.tokens)
}

/// The produced-token counter and the set it fills.
#[derive(Default)]
struct Sink {
    tokens: BTreeSet<String>,
    produced: usize,
}

impl Sink {
    fn push(&mut self, token: String) -> Result<(), TooManyTokens> {
        self.produced += 1;
        if self.produced > MAX_TOKENS {
            return Err(TooManyTokens {
                produced: self.produced,
            });
        }
        self.tokens.insert(token);
        Ok(())
    }
}

/// What the coarse tokens are computed from.
#[derive(Default)]
struct Shape {
    links: usize,
    caps_runs: usize,
    non_ascii: usize,
    non_space: usize,
}

/// One text, word by word, every token prefixed with `prefix`.
fn words(text: &str, prefix: &str, sink: &mut Sink) -> Result<Shape, TooManyTokens> {
    let mut shape = Shape::default();
    let mut in_caps_run = false;
    for raw in text.split_whitespace() {
        for c in raw.chars() {
            shape.non_space += 1;
            if !c.is_ascii() {
                shape.non_ascii += 1;
            }
        }
        let trimmed = raw.trim_matches(|c: char| c.is_ascii_punctuation());
        // A URL may open after punctuation, `(https://…)`; its own trailing punctuation is
        // trimmed by `url_tokens`, which keeps a closing `/`.
        let lowered_url = raw
            .trim_start_matches(|c: char| c.is_ascii_punctuation())
            .to_lowercase();
        if is_url(&lowered_url) {
            shape.links += 1;
            in_caps_run = false;
            for token in url_tokens(&lowered_url) {
                sink.push(format!("{prefix}{token}"))?;
            }
            continue;
        }
        if let Some(domain) = address_domain(trimmed) {
            in_caps_run = false;
            sink.push(format!("{prefix}addr:{domain}"))?;
            continue;
        }
        if is_all_caps(trimmed) {
            if !in_caps_run {
                shape.caps_runs += 1;
            }
            in_caps_run = true;
        } else {
            in_caps_run = false;
        }
        let word = trimmed.to_lowercase();
        let len = word.chars().count();
        if len < MIN_WORD {
            continue;
        }
        if len > MAX_WORD {
            let first = word
                .chars()
                .next()
                .expect("a long word has a first character");
            sink.push(format!("{prefix}skip:{first} {}", len / 10 * 10))?;
        } else {
            sink.push(format!("{prefix}{word}"))?;
        }
    }
    Ok(shape)
}

fn is_url(lowered: &str) -> bool {
    lowered.starts_with("http://") || lowered.starts_with("https://") || lowered.starts_with("www.")
}

/// A word is all-caps when it has at least three letters and every letter is uppercase.
fn is_all_caps(word: &str) -> bool {
    let mut letters = 0;
    for c in word.chars().filter(|c| c.is_alphabetic()) {
        if !c.is_uppercase() {
            return false;
        }
        letters += 1;
    }
    letters >= 3
}

/// The three tokens of a lowercased URL.
fn url_tokens(lowered: &str) -> [String; 3] {
    let trimmed = lowered.trim_end_matches(|c: char| c.is_ascii_punctuation() && c != '/');
    let (scheme, rest) = match trimmed.split_once("://") {
        Some((scheme, rest)) => (scheme, rest),
        None => ("none", trimmed),
    };
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    // Userinfo is dropped (it is per-message, and `user@host` is a classic disguise), and so is
    // the port.
    let host_port = authority.rsplit('@').next().unwrap_or("");
    let host = match host_port.rsplit_once(':') {
        Some((host, port)) if port.chars().all(|c| c.is_ascii_digit()) => host,
        _ => host_port,
    };
    let host = host.trim_end_matches('.');
    let (host, domain) = if is_ipv4(host) {
        ("ip-literal".to_string(), "ip-literal".to_string())
    } else if host.is_empty()
        || host.len() > MAX_HOST
        || !host
            .chars()
            .all(|c| c.is_alphanumeric() || c == '-' || c == '.')
    {
        ("malformed".to_string(), "malformed".to_string())
    } else {
        (host.to_string(), registrable(host))
    };
    [
        format!("url-scheme:{scheme}"),
        format!("url-host:{host}"),
        format!("url-domain:{domain}"),
    ]
}

fn is_ipv4(host: &str) -> bool {
    let parts: Vec<&str> = host.split('.').collect();
    parts.len() == 4
        && parts
            .iter()
            .all(|p| !p.is_empty() && p.len() <= 3 && p.chars().all(|c| c.is_ascii_digit()))
}

/// The registrable domain of a host, by the approximation in the module documentation.
///
/// ```
/// use ikigai_bayes::tokenize::registrable;
///
/// assert_eq!(registrable("www.shop.example.co.uk"), "example.co.uk");
/// assert_eq!(registrable("mail.example.com"), "example.com");
/// assert_eq!(registrable("example.com"), "example.com");
/// assert_eq!(registrable("localhost"), "localhost");
/// ```
pub fn registrable(host: &str) -> String {
    let labels: Vec<&str> = host.split('.').collect();
    let n = labels.len();
    if n <= 2 {
        return host.to_string();
    }
    let take = if labels[n - 1].len() == 2 && SECOND_LEVEL.contains(&labels[n - 2]) {
        3
    } else {
        2
    };
    labels[n - take..].join(".")
}

/// The lowercased domain of a word that is one email address, or `None`.
fn address_domain(word: &str) -> Option<String> {
    let (local, domain) = word.rsplit_once('@')?;
    let domain = domain.trim_end_matches('.').to_lowercase();
    let ok = !local.is_empty()
        && !local.contains(char::is_whitespace)
        && domain.contains('.')
        && !domain.starts_with('.')
        && domain.len() <= MAX_HOST
        && domain
            .chars()
            .all(|c| c.is_alphanumeric() || c == '-' || c == '.');
    ok.then_some(domain)
}

fn links_bucket(n: usize) -> &'static str {
    match n {
        0 => "0",
        1 => "1",
        2 => "2",
        3..=5 => "3-5",
        _ => "6+",
    }
}

fn caps_bucket(n: usize) -> &'static str {
    match n {
        0 => "0",
        1 => "1",
        2..=3 => "2-3",
        _ => "4+",
    }
}

fn non_ascii_bucket(non_ascii: usize, total: usize) -> &'static str {
    if non_ascii == 0 || total == 0 {
        return "0";
    }
    // Integer percent, rounded down; a single non-ASCII character in a long message is still
    // `1-9`, never `0`, which is reserved for "none at all".
    match (non_ascii * 100 / total).max(1) {
        1..=9 => "1-9",
        10..=29 => "10-29",
        30..=59 => "30-59",
        _ => "60-100",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(body: &str) -> Vec<String> {
        tokenize(body, &[]).unwrap().into_iter().collect()
    }

    fn has(body: &str, token: &str) -> bool {
        tokenize(body, &[]).unwrap().contains(token)
    }

    #[test]
    fn words_are_lowercased_trimmed_and_windowed() {
        let t = toks("Hello, WORLD! an ok-ish \"quoted\" internationalization");
        for want in ["hello", "world", "ok-ish", "quoted", "internationalization"] {
            assert!(t.contains(&want.to_string()), "{want} in {t:?}");
        }
        // Two characters: dropped.
        assert!(!t.contains(&"an".to_string()));
        assert!(!t.contains(&"ok".to_string()));
    }

    #[test]
    fn a_long_word_becomes_a_skip_token() {
        let word = "a".repeat(37);
        assert!(has(&word, "skip:a 30"), "{:?}", toks(&word));
        // Exactly MAX_WORD is kept as itself.
        assert!(has(&"b".repeat(MAX_WORD), &"b".repeat(MAX_WORD)));
        assert!(has(&"c".repeat(MAX_WORD + 1), "skip:c 20"));
    }

    #[test]
    fn a_url_is_scheme_host_and_registrable_domain() {
        let t = toks("see https://User@WWW.Example.co.uk:8443/a/b?x=1#f.");
        for want in [
            "url-scheme:https",
            "url-host:www.example.co.uk",
            "url-domain:example.co.uk",
            "links:1",
        ] {
            assert!(t.contains(&want.to_string()), "{want} in {t:?}");
        }
        // The path and query are dropped.
        assert!(!t.iter().any(|x| x.contains("x=1")), "{t:?}");
    }

    #[test]
    fn a_url_after_punctuation_is_still_a_url() {
        let t = toks("(see https://example.com/x).");
        assert!(t.contains(&"url-domain:example.com".to_string()), "{t:?}");
        assert!(t.contains(&"links:1".to_string()), "{t:?}");
    }

    #[test]
    fn a_bare_www_url_has_no_scheme() {
        let t = toks("www.example.com/path");
        assert!(t.contains(&"url-scheme:none".to_string()), "{t:?}");
        assert!(t.contains(&"url-domain:example.com".to_string()), "{t:?}");
    }

    #[test]
    fn an_ip_literal_host_is_its_own_signal() {
        let t = toks("http://192.0.2.7/login");
        assert!(t.contains(&"url-host:ip-literal".to_string()), "{t:?}");
        assert!(t.contains(&"url-domain:ip-literal".to_string()), "{t:?}");
    }

    #[test]
    fn a_malformed_host_is_one_token() {
        assert!(has("http://exa_mple.com/", "url-host:malformed"));
        let long = format!("http://{}.com/", "a".repeat(260));
        assert!(has(&long, "url-host:malformed"));
    }

    #[test]
    fn an_address_is_its_domain_only() {
        let t = toks("write to someone@Mail.Example.org today");
        assert!(t.contains(&"addr:mail.example.org".to_string()), "{t:?}");
        assert!(!t.iter().any(|x| x.contains("someone")), "{t:?}");
    }

    #[test]
    fn coarse_link_buckets() {
        let urls = |n: usize| {
            (0..n)
                .map(|i| format!("http://h{i}.example.com "))
                .collect::<String>()
        };
        assert!(has("plain words", "links:0"));
        assert!(has(&urls(1), "links:1"));
        assert!(has(&urls(2), "links:2"));
        assert!(has(&urls(4), "links:3-5"));
        assert!(has(&urls(9), "links:6+"));
    }

    #[test]
    fn coarse_caps_runs_count_runs_not_words() {
        assert!(has("all quiet here", "caps-runs:0"));
        // One run of three all-caps words.
        assert!(has("BUY NOW CHEAP please", "caps-runs:1"));
        // Two runs, broken by a lowercase word; `OK` is too short to count.
        assert!(has("FREE money FREE OK", "caps-runs:2-3"));
        assert!(has("AAA b CCC d EEE f GGG", "caps-runs:4+"));
    }

    #[test]
    fn coarse_non_ascii_share() {
        assert!(has("plain ascii", "non-ascii:0"));
        assert!(has(
            "naïve and plenty of ascii text around it",
            "non-ascii:1-9"
        ));
        assert!(has("ééééé abcde abcde", "non-ascii:30-59"));
        assert!(has("ééééééééé", "non-ascii:60-100"));
    }

    #[test]
    fn fields_are_prefixed_and_addresses_reduce_to_their_domain() {
        let t: Vec<String> = tokenize(
            "hello there",
            &[
                ("subject", "Cheap PILLS"),
                ("email", "someone@example.net"),
                ("from", "not an address"),
                ("organisation", "Example Widgets"),
            ],
        )
        .unwrap()
        .into_iter()
        .collect();
        for want in [
            "subject:cheap",
            "subject:pills",
            "email:example.net",
            "from:no-address",
            "from:not",
            "from:address",
            "organisation:example",
            "organisation:widgets",
            "hello",
        ] {
            assert!(t.contains(&want.to_string()), "{want} in {t:?}");
        }
        // Fields get no coarse tokens of their own.
        assert!(!t.iter().any(|x| x.starts_with("subject:links:")), "{t:?}");
    }

    #[test]
    fn the_answer_is_a_set() {
        let t = tokenize("spam spam spam spam", &[]).unwrap();
        assert_eq!(t.iter().filter(|x| *x == "spam").count(), 1);
    }

    #[test]
    fn too_many_tokens_is_refused_not_truncated() {
        let body = "abc ".repeat(MAX_TOKENS);
        // MAX_TOKENS words plus three coarse tokens.
        let err = tokenize(&body, &[]).unwrap_err();
        assert_eq!(err.produced, MAX_TOKENS + 1);
        assert!(err.to_string().contains("refused, not truncated"));
        // Exactly at the bound (MAX_TOKENS - 3 words + 3 coarse) is accepted.
        assert!(tokenize(&"abc ".repeat(MAX_TOKENS - 3), &[]).is_ok());
    }

    #[test]
    fn the_version_is_pinned() {
        // Changing any rule above is a version bump; this line is the reminder.
        assert_eq!(TOKENIZER_VERSION, "1");
    }
}
