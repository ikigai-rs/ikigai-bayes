//! The resources: one door per model and part, each declaring the capability it enforces.
//!
//! Every door is bound EXACT (`urn:bayes:contact:score`, not a template), because the host names
//! the models when it opens them. That is what lets each door declare its model's own capability
//! exactly (`urn:cap:bayes:read:contact`) and have the kernel enforce it: there is no runtime gate
//! in this module, so what is declared is all that is enforced.

use std::sync::Arc;

use ikigai_core::{
    ArgSpec, ContentId, Description, EndpointSpace, Error, Exact, FnEndpoint, Invocation, ReprType,
    Representation, Result, Verb,
};
use serde_json::json;

use crate::model::{Change, Label, Labels, Score};
use crate::store::Store;
use crate::tokenize::{FIELDS, TOKENIZER_VERSION};
use crate::{
    model_iri, read_cap, too_many, train, train_cap, untrain, Bayes, Message, Slot,
    DEFAULT_CLUES_SHOWN, MAX_MESSAGE_BYTES,
};

const XSD: &str = "http://www.w3.org/2001/XMLSchema#";

const TEXT: &str = "text/plain";
const JSON: &str = "application/json";
// Every `json!` object below lists its keys in sorted order, so a JSON face is byte-identical
// whether or not some other crate in a host's graph turns on serde_json's `preserve_order`
// (features unify across the graph, and without it serde_json sorts keys).
const TURTLE: &str = "text/turtle";

/// The faces of a score.
const SCORE_FACES: [&str; 3] = [TEXT, JSON, TURTLE];
/// The faces of everything else.
const PLAIN_FACES: [&str; 2] = [TEXT, JSON];

fn xsd(local: &str) -> String {
    format!("{XSD}{local}")
}

/// The space: every model's four doors, then `urn:bayes:models`. It is built from the models the
/// host opened, so the HOST names it (a name is a claim that every space so named holds the same
/// doors, and only the host knows which models it passed in).
///
/// ```
/// use ikigai_bayes::{Bayes, ModelConfig};
/// use ikigai_core::Space;
///
/// let bayes = Bayes::in_memory([ModelConfig::new("contact")]).unwrap();
/// assert!(ikigai_bayes::space(&bayes).id().is_none());
/// ```
pub fn space(bayes: &Bayes) -> EndpointSpace {
    let mut space = EndpointSpace::new();
    for slot in bayes.slots() {
        let name = &slot.name;
        space = space
            .bind(Exact::new(model_iri(name, "score")), score(slot.clone()))
            .bind(
                Exact::new(model_iri(name, "train")),
                train_door(bayes.store().clone(), slot.clone()),
            )
            .bind(
                Exact::new(model_iri(name, "untrain")),
                untrain_door(bayes.store().clone(), slot.clone()),
            )
            .bind(Exact::new(model_iri(name, "stats")), stats(slot.clone()));
    }
    let names: Vec<String> = bayes.models().map(String::from).collect();
    space.bind(Exact::new("urn:bayes:models"), models(names))
}

// --- arguments ----------------------------------------------------------------------------

fn optional_str<'a>(inv: &'a Invocation<'_>, name: &str) -> Result<Option<&'a str>> {
    match inv.request.args.get(name) {
        None => Ok(None),
        Some(_) => inv.inline_str(name).map(Some),
    }
}

/// The face asked for with `as=`, or the first.
fn face<'a>(inv: &Invocation<'_>, faces: &[&'a str]) -> Result<&'a str> {
    match optional_str(inv, "as")? {
        None => Ok(faces[0]),
        Some(asked) => {
            faces
                .iter()
                .copied()
                .find(|f| *f == asked)
                .ok_or_else(|| Error::InvalidArgument {
                    name: "as".into(),
                    detail: format!("`{asked}` is not one of {}", faces.join(", ")),
                })
        }
    }
}

/// The message a call carries: `content` (required unless `required` is false and it is
/// absent) plus any declared field, bounded by [`MAX_MESSAGE_BYTES`].
fn message(inv: &Invocation<'_>) -> Result<Message> {
    let content = inv.inline_str("content")?;
    let mut message = Message::new(content);
    for field in FIELDS {
        if let Some(value) = optional_str(inv, field)? {
            message = message
                .field(field, value)
                .map_err(|detail| Error::InvalidArgument {
                    name: field.into(),
                    detail,
                })?;
        }
    }
    if message.len() > MAX_MESSAGE_BYTES {
        return Err(Error::InvalidArgument {
            name: "content".into(),
            detail: format!(
                "the message is {} bytes (body and fields), past the bound of \
                 {MAX_MESSAGE_BYTES}: refused, not truncated",
                message.len()
            ),
        });
    }
    Ok(message)
}

fn content_spec(summary: &str) -> ArgSpec {
    ArgSpec::new("content")
        .summary(summary)
        .class(xsd("string"))
}

fn field_specs() -> impl Iterator<Item = ArgSpec> {
    FIELDS.into_iter().map(|field| {
        let what = match field {
            "subject" => "the message's subject",
            "from" => "the sender's address (tokenized as its domain)",
            "name" => "the sender's name",
            "email" => "the sender's email address (tokenized as its domain)",
            _ => "the sender's organization",
        };
        ArgSpec::new(field)
            .summary(format!(
                "{what}, tokenized under the prefix `{field}:`; part of the message's digest"
            ))
            .class(xsd("string"))
            .optional()
    })
}

fn as_spec(faces: &[&str]) -> ArgSpec {
    ArgSpec::new("as")
        .summary("the representation to answer with")
        .class(xsd("string"))
        .one_of(faces.iter().copied())
        .default_value(faces[0])
}

fn with_inputs(mut d: Description, inputs: impl IntoIterator<Item = ArgSpec>) -> Description {
    for input in inputs {
        d = d.input(input);
    }
    d
}

fn with_outputs(mut d: Description, faces: &[&str]) -> Description {
    for f in faces {
        d = d.output(*f);
    }
    d
}

fn answer(face: &str, body: String) -> Representation {
    let repr_type = if face == JSON {
        ReprType::new(JSON)
    } else {
        ReprType::new(face).with_param("charset", "utf-8")
    };
    Representation::new(repr_type, body.into_bytes())
}

/// Hang a model read from both of its write doors' threads: the kernel cuts each after a write
/// through it.
fn hang_from_model(inv: &Invocation<'_>, model: &str) {
    inv.depends_on(model_iri(model, "train"));
    inv.depends_on(model_iri(model, "untrain"));
}

// Keep dynamic label keys sorted, including with serde_json's preserve_order feature.
fn named_pair<T: serde::Serialize>(
    labels: &Labels,
    high: T,
    low: T,
) -> std::collections::BTreeMap<&str, T> {
    [
        (labels.name(Label::Spam), high),
        (labels.name(Label::Ham), low),
    ]
    .into_iter()
    .collect()
}

// --- score --------------------------------------------------------------------------------

/// The most clues `clues=` may ask to be shown: every clue a score can combine.
const MAX_CLUES_SHOWN: usize = 150;

fn score(slot: Arc<Slot>) -> FnEndpoint {
    let name = slot.name.clone();
    let describe = with_outputs(
        with_inputs(
            Description::new(format!("bayes-{name}-score"))
                .title(format!("Score a message against the `{name}` model"))
                .summary(
                    "Scores a message with Robinson's per-token probabilities combined by \
                     Fisher's chi-squared method (SpamBayes): a score in [0, 1], its class \
                     (the high label at or above its cutoff, the low label below its cutoff, `unsure` \
                     between) and the strongest clues. Cacheable until the model is trained or \
                     untrained.",
                )
                .verb(Verb::Source)
                .verb(Verb::Meta)
                .requires(read_cap(&name)),
            [content_spec("the message text (piped)")]
                .into_iter()
                .chain(field_specs())
                .chain([
                    ArgSpec::new("clues")
                        .summary(format!(
                            "how many of the strongest clues to show, at most {MAX_CLUES_SHOWN} \
                             (the score always combines every clue)"
                        ))
                        .class(xsd("nonNegativeInteger"))
                        .default_value(DEFAULT_CLUES_SHOWN.to_string()),
                    as_spec(&SCORE_FACES),
                ]),
        ),
        &SCORE_FACES,
    );
    FnEndpoint::new(format!("bayes-{name}-score"), move |inv| {
        let face = face(inv, &SCORE_FACES)?;
        let message = message(inv)?;
        let shown = match optional_str(inv, "clues")? {
            None => DEFAULT_CLUES_SHOWN,
            Some(text) => text
                .trim()
                .parse::<usize>()
                .ok()
                .filter(|n| *n <= MAX_CLUES_SHOWN)
                .ok_or_else(|| Error::InvalidArgument {
                    name: "clues".into(),
                    detail: format!("`{text}` is not an integer from 0 to {MAX_CLUES_SHOWN}"),
                })?,
        };
        let tokens = message.tokens().map_err(too_many)?;
        let digest = message.digest();
        let score = slot
            .model
            .lock()
            .expect("model lock")
            .score(&tokens, &slot.settings);
        hang_from_model(inv, &slot.name);
        let body = match face {
            JSON => score_json(&slot, &digest, &score, shown),
            TURTLE => score_turtle(&slot, &digest, &score, shown),
            _ => score_text(&slot, &digest, &score, shown),
        };
        Ok(answer(face, body).cacheable())
    })
    .with_description(describe)
}

/// The text face: `<class> <score>`, the digest, `clues <shown> of <combined>`, then one
/// `<probability> <token>` line per clue shown, strongest first.
fn score_text(slot: &Slot, digest: &ContentId, score: &Score, shown: usize) -> String {
    let mut out = format!(
        "{} {:.6}\ndigest {digest}\nmodelDigest {}\nclues {} of {}\n",
        slot.labels.class(score.class),
        score.score,
        score.model_digest,
        shown.min(score.clues.len()),
        score.clues.len()
    );
    for clue in score.clues.iter().take(shown) {
        out.push_str(&format!("{:.6} {}\n", clue.probability, clue.token));
    }
    out
}

fn score_json(slot: &Slot, digest: &ContentId, score: &Score, shown: usize) -> String {
    let clues: Vec<_> = score
        .clues
        .iter()
        .take(shown)
        .map(|c| json!({"probability": c.probability, "token": c.token}))
        .collect();
    json!({
        "class": slot.labels.class(score.class),
        "clueCount": score.clues.len(),
        "clues": clues,
        "cutoffs": named_pair(&slot.labels, slot.settings.spam_cutoff, slot.settings.ham_cutoff),
        "digest": digest.to_string(),
        "model": slot.name,
        "modelDigest": score.model_digest.to_string(),
        "score": score.score,
        "tokenizer": TOKENIZER_VERSION,
    })
    .to_string()
}

/// The Turtle face, skolemized: the score is `urn:bayes:{model}:score:{digest}`, each clue shown
/// is that IRI plus `:clue:{percent-encoded token}`. Only well-known vocabularies: schema.org's
/// `Rating` and `PropertyValue`, PROV, Dublin Core terms.
fn score_turtle(slot: &Slot, digest: &ContentId, score: &Score, shown: usize) -> String {
    let subject = format!("{}:{digest}", model_iri(&slot.name, "score"));
    let mut out = String::from(
        "@prefix schema: <https://schema.org/> .\n\
         @prefix prov: <http://www.w3.org/ns/prov#> .\n\
         @prefix dcterms: <http://purl.org/dc/terms/> .\n\
         @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n\n",
    );
    out.push_str(&format!(
        "<{subject}> a schema:Rating ;\n  \
         schema:ratingValue \"{:.6}\"^^xsd:decimal ;\n  \
         schema:worstRating \"0\"^^xsd:decimal ;\n  \
         schema:bestRating \"1\"^^xsd:decimal ;\n  \
         schema:category \"{}\" ;\n  \
         schema:ratingExplanation \"{} below {}, {} at or above {}; {} clue(s) combined\" ;\n  \
         dcterms:identifier \"{digest}\" ;\n  \
         prov:wasDerivedFrom <{}>",
        score.score,
        slot.labels.class(score.class),
        slot.labels.name(Label::Ham),
        slot.settings.ham_cutoff,
        slot.labels.name(Label::Spam),
        slot.settings.spam_cutoff,
        score.clues.len(),
        model_iri(&slot.name, "stats"),
    ));
    out.push_str(&format!(" ;\n  schema:version \"{}\"", score.model_digest));
    let clues: Vec<_> = score.clues.iter().take(shown).collect();
    let iri = |token: &str| format!("{subject}:clue:{}", encode(token));
    if !clues.is_empty() {
        out.push_str(" ;\n  schema:additionalProperty ");
        out.push_str(
            &clues
                .iter()
                .map(|c| format!("<{}>", iri(&c.token)))
                .collect::<Vec<_>>()
                .join(", "),
        );
    }
    out.push_str(" .\n");
    for clue in clues {
        out.push_str(&format!(
            "\n<{}> a schema:PropertyValue ;\n  schema:name \"{}\" ;\n  schema:value \"{:.6}\"^^xsd:decimal .\n",
            iri(&clue.token),
            escape(&clue.token),
            clue.probability
        ));
    }
    out
}

/// Percent-encode every byte outside the IRI-unreserved set, so a token of any characters is one
/// IRI segment.
fn encode(token: &str) -> String {
    let mut out = String::with_capacity(token.len());
    for b in token.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// A Turtle string literal's escapes.
fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04X}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

// --- train --------------------------------------------------------------------------------

fn train_door(store: Arc<dyn Store>, slot: Arc<Slot>) -> FnEndpoint {
    let name = slot.name.clone();
    let describe = with_outputs(
        with_inputs(
            Description::new(format!("bayes-{name}-train"))
                .title(format!("Teach the `{name}` model a message"))
                .summary(
                    "Trains the model on a message with a label. Idempotent per message (keyed by \
                     its digest): the same message twice counts once, and the same message with \
                     the other label MOVES, which is how a correction works. Answers what \
                     changed: added, moved or unchanged.",
                )
                .verb(Verb::Sink)
                .verb(Verb::Meta)
                .requires(train_cap(&name)),
            [
                content_spec("the message text (piped, or the sink's body)"),
                ArgSpec::new("label")
                    .summary("what the message is")
                    .class(xsd("string"))
                    .one_of([slot.labels.name(Label::Spam), slot.labels.name(Label::Ham)]),
            ]
            .into_iter()
            .chain(field_specs())
            .chain([as_spec(&PLAIN_FACES)]),
        ),
        &PLAIN_FACES,
    );
    FnEndpoint::new(format!("bayes-{name}-train"), move |inv| {
        let face = face(inv, &PLAIN_FACES)?;
        let message = message(inv)?;
        let label_text = inv.inline_str("label")?;
        let label = slot
            .labels
            .parse(label_text.trim())
            .ok_or_else(|| Error::InvalidArgument {
                name: "label".into(),
                detail: format!(
                    "`{label_text}` is not `{}` or `{}`",
                    slot.labels.name(Label::Spam),
                    slot.labels.name(Label::Ham)
                ),
            })?;
        let done = train(store.as_ref(), &slot, &message, label)?;
        let (change, previous) = match done.change {
            Change::Added => ("added", None),
            Change::Moved { from } => ("moved", Some(slot.labels.name(from))),
            Change::Unchanged => ("unchanged", None),
        };
        let body = if face == JSON {
            json!({
                "change": change,
                "digest": done.digest.to_string(),
                "label": slot.labels.name(done.label),
                "messages": named_pair(&slot.labels, done.messages.0, done.messages.1),
                "model": slot.name,
                "previous": previous,
            })
            .to_string()
        } else {
            let what = match done.change {
                Change::Added => {
                    format!("added {} as {}", done.digest, slot.labels.name(done.label))
                }
                Change::Moved { from } => format!(
                    "moved {} from {} to {}",
                    done.digest,
                    slot.labels.name(from),
                    slot.labels.name(done.label)
                ),
                Change::Unchanged => {
                    format!(
                        "unchanged {}: already {}",
                        done.digest,
                        slot.labels.name(done.label)
                    )
                }
            };
            format!(
                "{what}\nmessages {} {} {} {}\n",
                slot.labels.name(Label::Spam),
                done.messages.0,
                slot.labels.name(Label::Ham),
                done.messages.1
            )
        };
        Ok(answer(face, body))
    })
    .with_description(describe)
}

// --- untrain ------------------------------------------------------------------------------

fn untrain_door(store: Arc<dyn Store>, slot: Arc<Slot>) -> FnEndpoint {
    let name = slot.name.clone();
    let describe = with_outputs(
        with_inputs(
            Description::new(format!("bayes-{name}-untrain"))
                .title(format!("Make the `{name}` model forget a message"))
                .summary(
                    "Removes a trained message, given as `content` (with the same fields it was \
                     trained with) or as `digest=`, exactly one of the two. The model is then \
                     exactly as if it had never been taught it. A message never trained is \
                     NotFound.",
                )
                .verb(Verb::Delete)
                .verb(Verb::Meta)
                .requires(train_cap(&name)),
            [
                content_spec(
                    "the message text (piped, or the delete's body); give this or digest=",
                )
                .optional(),
                ArgSpec::new("digest")
                    .summary(
                        "the message's digest as train and score report it (`b3:<hex>`); give \
                         this or content",
                    )
                    .class(xsd("string"))
                    .optional(),
            ]
            .into_iter()
            .chain(field_specs())
            .chain([as_spec(&PLAIN_FACES)]),
        ),
        &PLAIN_FACES,
    );
    FnEndpoint::new(format!("bayes-{name}-untrain"), move |inv| {
        let face = face(inv, &PLAIN_FACES)?;
        let has_content = inv.request.args.contains_key("content");
        let digest = match (has_content, optional_str(inv, "digest")?) {
            (true, Some(_)) => {
                return Err(Error::InvalidArgument {
                    name: "digest".into(),
                    detail: "give content or digest=, not both".into(),
                })
            }
            (false, None) => return Err(Error::MissingArgument("content".into())),
            (true, None) => message(inv)?.digest(),
            (false, Some(text)) => {
                ContentId::parse(text.trim()).map_err(|e| Error::InvalidArgument {
                    name: "digest".into(),
                    detail: e.to_string(),
                })?
            }
        };
        let done = untrain(store.as_ref(), &slot, digest)?;
        let body = if face == JSON {
            json!({
                "digest": done.digest.to_string(),
                "messages": named_pair(&slot.labels, done.messages.0, done.messages.1),
                "model": slot.name,
                "removed": slot.labels.name(done.label),
            })
            .to_string()
        } else {
            format!(
                "removed {} (was {})\nmessages {} {} {} {}\n",
                done.digest,
                slot.labels.name(done.label),
                slot.labels.name(Label::Spam),
                done.messages.0,
                slot.labels.name(Label::Ham),
                done.messages.1
            )
        };
        Ok(answer(face, body))
    })
    .with_description(describe)
}

// --- stats --------------------------------------------------------------------------------

fn stats(slot: Arc<Slot>) -> FnEndpoint {
    let name = slot.name.clone();
    let describe = with_outputs(
        with_inputs(
            Description::new(format!("bayes-{name}-stats"))
                .title(format!("The `{name}` model's counts"))
                .summary(
                    "The model's trained messages by label, its distinct tokens, its \
                     cutoffs and the tokenizer version it was trained with. Cacheable until the \
                     model is trained or untrained.",
                )
                .verb(Verb::Source)
                .verb(Verb::Meta)
                .requires(read_cap(&name)),
            [as_spec(&PLAIN_FACES)],
        ),
        &PLAIN_FACES,
    );
    FnEndpoint::new(format!("bayes-{name}-stats"), move |inv| {
        let face = face(inv, &PLAIN_FACES)?;
        let (nspam, nham, tokens, model_digest) = {
            let model = slot.model.lock().expect("model lock");
            (model.nspam(), model.nham(), model.distinct_tokens(), model.digest())
        };
        hang_from_model(inv, &slot.name);
        let s = &slot.settings;
        let body = if face == JSON {
            json!({
                "cutoffs": named_pair(&slot.labels, s.spam_cutoff, s.ham_cutoff),
                "messages": named_pair(&slot.labels, nspam, nham),
                "model": slot.name,
                "modelDigest": model_digest.to_string(),
                "tokenizer": TOKENIZER_VERSION,
                "tokens": tokens,
            })
            .to_string()
        } else {
            format!(
                "model {}\nmessages {} {nspam} {} {nham}\ntokens {tokens}\ncutoffs {} {} {} {}\ntokenizer {TOKENIZER_VERSION}\nmodelDigest {model_digest}\n",
                slot.name, slot.labels.name(Label::Spam), slot.labels.name(Label::Ham), slot.labels.name(Label::Ham), s.ham_cutoff, slot.labels.name(Label::Spam), s.spam_cutoff
            )
        };
        Ok(answer(face, body).cacheable())
    })
    .with_description(describe)
}

// --- models -------------------------------------------------------------------------------

fn models(names: Vec<String>) -> FnEndpoint {
    let describe = with_outputs(
        Description::new("bayes-models")
            .title("The classifier models this host offers")
            .summary(
                "The models this host offers, one name per line (the `..` map convention), and \
                 with `as=application/json` the capability each part of each one requires. Fixed \
                 for the life of the host: a host offers other models by opening again.",
            )
            .verb(Verb::Source)
            .verb(Verb::Meta)
            .input(as_spec(&PLAIN_FACES)),
        &PLAIN_FACES,
    );
    FnEndpoint::new("bayes-models", move |inv| {
        let face = face(inv, &PLAIN_FACES)?;
        let body = if face == JSON {
            let models: Vec<_> = names
                .iter()
                .map(|n| {
                    json!({
                        "name": n,
                        "readCapability": read_cap(n),
                        "score": model_iri(n, "score"),
                        "stats": model_iri(n, "stats"),
                        "train": model_iri(n, "train"),
                        "trainCapability": train_cap(n),
                        "untrain": model_iri(n, "untrain"),
                    })
                })
                .collect();
            json!({ "models": models }).to_string()
        } else {
            names.join("\n")
        };
        Ok(answer(face, body).cacheable())
    })
    .with_description(describe)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_keeps_unreserved_and_escapes_the_rest() {
        assert_eq!(encode("url-domain:example.com"), "url-domain%3Aexample.com");
        assert_eq!(encode("skip:a 30"), "skip%3Aa%2030");
        assert_eq!(encode("é"), "%C3%A9");
    }

    #[test]
    fn escape_makes_a_turtle_string() {
        assert_eq!(escape("a\"b\\c\nd"), "a\\\"b\\\\c\\nd");
    }
}
