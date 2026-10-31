# ikigai-bayes

A spam filter that learns from you, as [ikigai](https://github.com/ikigai-rs) resources.

A contact form with a honeypot still gets spam: what gets past a honeypot is not a naive bot.
`ikigai-bayes` is a naive-Bayes text classifier in the SpamBayes style that a host serves as
resources. You score each message, you tell it when it was wrong, and it learns. It is a general
module, not a contact-form hack: any text with a two-way label (spam and ham, wanted and not) is
the same problem.

```text
sink   urn:bayes:contact:train  label=spam      <- a message you marked as spam
sink   urn:bayes:contact:train  label=ham       <- one you marked as fine (a correction MOVES it)
source urn:bayes:contact:score                  <- every new message: spam / ham / unsure, and why
```

## Why Bayes, and why this kind

- **It learns from your labels, not someone's list.** What counts as spam on one form is
  ordinary mail on another. A Bayesian filter is trained on exactly the messages you judged.
- **It says *unsure*.** Fisher's combining judges the evidence for spam and for ham separately,
  so a message with strong evidence both ways, or none, scores near 0.5 instead of being forced
  to a side. Two cutoffs, not one threshold, make three classes. Shadow mode (score everything,
  change nothing) and quarantine (move only confident spam aside) both depend on that middle band.
- **It explains itself.** Every score lists its strongest clues, each token with its
  probability, so a wrong call can be read rather than guessed at.
- **No ML dependency.** The method is a page of arithmetic (below), and this crate is that page.

## The resources

One model per name; the HOST names the models when it opens them. Each model is four resources,
and each declares exactly the capability the kernel enforces:

| resource | verb | answer | capability |
|---|---|---|---|
| `urn:bayes:{model}:score` | Source | the score, the class (`spam` / `ham` / `unsure`) and the strongest clues | `urn:cap:bayes:read:{model}` |
| `urn:bayes:{model}:train` | Sink | what changed: `added`, `moved` or `unchanged` | `urn:cap:bayes:train:{model}` |
| `urn:bayes:{model}:untrain` | Delete | what was removed | `urn:cap:bayes:train:{model}` |
| `urn:bayes:{model}:stats` | Source | message and token counts, the cutoffs, the tokenizer version | `urn:cap:bayes:read:{model}` |
| `urn:bayes:models` | Source | the models this host offers (one name per line) | none |

- **`score`** takes the message as `content` (required, so a pipe fills it) and, optionally, the
  named fields `subject`, `from`, `name`, `email` and `organisation` (the contact form's existing
  field name), each tokenized under its own prefix. `clues=` sets how many clues are shown
  (default 15; the score always combines all of them). Faces: `text/plain` (default),
  `as=application/json`, and `as=text/turtle`, which is skolemized (no blank nodes) and uses only
  schema.org, PROV and Dublin Core terms.
- **`train`** takes `content`, the same optional fields, and `label` (`spam` or `ham`). It is
  **idempotent per message**: a message is keyed by its digest, so training it twice counts once,
  and training it with the other label MOVES it, which is how a correction works.
- **`untrain`** takes the message as `content` (with the fields it was trained with) or as
  `digest=` (`b3:<hex>`, as train and score report it), and leaves the model **exactly** as if it
  had never been taught it. A message never trained is `NotFound`.

Every answer's exact shape is a doctest in `src/lib.rs`.

### Caching

Models default to `spam` (high score) and `ham` (low score). For another question,
use `ModelConfig::new("announcements").labels("yes", "no")?`. The first label is
at or above the high cutoff; the second is below the low cutoff. `unsure` remains
the class between them. Label names are distinct lowercase ASCII names, using
letters, digits and single hyphens, up to 40 characters; `unsure` and `models` are
reserved. `cutoffs(low, high)` still sets the two thresholds.

Labels are declared in config and persisted in the model file to prevent a host
from silently reinterpreting existing training. Reopening must use the same
labels. Old files without labels retain `spam`/`ham`, and their serialization is
unchanged. Train's `one_of`, response labels, classes, count keys and cutoff keys
all use the declared names. Clue probabilities always point toward the first
(high-score) label.

Every score and stats answer includes a BLAKE3 trained-state digest: `modelDigest`
in JSON and text, and `schema:version` on Turtle scores. The message's existing
`digest` is separate. The state digest hashes deterministic model-file bytes with
the model name empty: tokenizer version, labels and messages sorted by message
digest (with sorted tokens). It excludes host name, cutoffs and rendering options.
Identical trained states share a digest across model names and training order.
Adding, moving or removing training changes it; idempotent training, failed writes,
reads and reopening preserve it. Exact untraining restores the previous digest.
Record cutoffs separately if a judgment also needs to reproduce the class.

A score and the stats are cacheable on their inputs and hang from the model's two golden threads,
`urn:bayes:{model}:train` and `urn:bayes:{model}:untrain`, which the kernel cuts after every write
through them. So a score is served from the cache until that model learns or forgets something,
and then recomputes; training another model cuts nothing. `tests/resources.rs` measures this with
`Kernel::is_cached` rather than reading it off the declarations.

## The method

The per-token probability is Robinson's, built on Graham's ratio. For a token `w` in `s` of
`nspam` spam messages and `h` of `nham` ham ones, with `n = s + h`:

```text
p(w) = (s / nspam) / (s / nspam + h / nham)          Graham
f(w) = (S·x + n·p(w)) / (S + n)                      Robinson, x = 0.5, S = 0.45
```

The tokens whose `f(w)` lies at least 0.1 from 0.5 are the clues, at most 150, strongest first.
Fisher's method combines them, judging each direction with the chi-squared survival function
`Q(χ², 2n)`:

```text
S = 1 − Q(−2·Σ ln(1 − f(w)), 2n)        evidence of spam
H = 1 − Q(−2·Σ ln f(w),       2n)        evidence of ham
score = (1 + S − H) / 2
```

A score below the **ham cutoff (0.20)** is ham, at or above the **spam cutoff (0.90)** is spam,
and between them is unsure. Every number above is SpamBayes's default; the cutoffs can be set per
model by the host. Unit tests check `f(w)`, `Q` and the combining against hand-computed values.

- Paul Graham, [*A Plan for Spam*](https://paulgraham.com/spam.html) (2002), and
  [*Better Bayesian Filtering*](https://paulgraham.com/better.html) (2003).
- Gary Robinson, *A Statistical Approach to the Spam Problem*, Linux Journal 107 (2003).
- The SpamBayes project: [background](https://spambayes.sourceforge.io/background.html) and the
  [classifier and tokenizer](https://github.com/smontanaro/spambayes) this crate follows.

### The tokenizer (version 1)

A tokenizer change invalidates a model, so the tokenizer is versioned, every model records the
version it was trained with, and a model trained with another version is refused at open. The
rules, each pinned by a test (`src/tokenize.rs` documents them in full):

- **Words**: split on whitespace, surrounding punctuation trimmed, lowercased; 3 to 20
  characters kept; a longer one becomes SpamBayes's `skip:<first char> <length/10·10>` token.
- **URLs** (`http://`, `https://`, `www.`): reduced to `url-scheme:`, `url-host:` and
  `url-domain:` (the registrable domain, by an approximation documented in the source rather than
  the Public Suffix List). An IP-literal host is its own token.
- **Email addresses**: `addr:<domain>`; the local part is dropped.
- **Coarse tokens**: `links:` (0, 1, 2, 3-5, 6+), `caps-runs:` (runs of ALL-CAPS words: 0, 1,
  2-3, 4+) and `non-ascii:` (percent of non-ASCII characters: 0, 1-9, 10-29, 30-59, 60-100).
- **Fields** are tokenized by the same rules under their own prefix (`subject:pills`); `from`
  and `email` reduce to their domain (`from:example.com`).

Each token counts once per message.

## Hosting it

```rust,ignore
let bayes = ikigai_bayes::Bayes::open(
    state_dir,                                        // the host's choice, e.g. from a flag
    [ikigai_bayes::ModelConfig::new("contact")],      // .cutoffs(ham, spam) to change them
)?;
let space = ikigai_bayes::space(&bayes);              // host-named: the host names it
```

The **state directory is the host's**: no environment variable, no hidden path under `$HOME`.

```text
<state>/LOCK              held for the host's life: one writer per directory
<state>/models            the registry: every model ever created, one per line
<state>/<model>.json      the model
```

- A model **is** the set of messages it was taught (digest, label, token set); the counts are
  derived from that set when it is read. So counts cannot drift, and untraining is exact.
- **Writes are atomic** (a temporary file, synced, renamed over the old one), and a train is
  answered only after its write lands; a failed write leaves memory and disk as they were.
- **Missing or corrupt is refused by name.** Only a NEW model (not in the registry, no file)
  starts empty. A registered model whose file is gone, a file that does not parse, another model's
  file, or one trained by another tokenizer version stops the host at open.
- ⚠ The state holds the token set of every trained message (that is what makes untraining by
  digest exact), so it is as sensitive as a bag of words of those messages. Keep it private.

`store::Store` is a small trait; the directory store is not built for `wasm32`, and the classifier,
the tokenizer and the in-memory store are (`Bayes::in_memory`). CI runs `clippy --lib` on
`wasm32-unknown-unknown`.

## Bounds

Every bound refuses; none truncates. A truncated message would be judged on a prefix nobody chose.

| what | bound | past it |
|---|---|---|
| a message (body plus every field) | 64 KiB | `InvalidArgument` on `content` |
| tokens one message produces | 16,384 | `InvalidArgument` on `content` |
| distinct tokens per model | 250,000 | train refused (`Conflict`) |
| messages per model | 50,000 | train of a NEW message refused (`Conflict`); moves still work |
| a model file | 64 MiB | train refused (`Conflict`); a larger file is refused at open |
| models per host | 64 | refused at open |
| a model name | 40 characters, `[a-z][a-z0-9-]*` | refused at open |
| clues shown (`clues=`) | 150 | `InvalidArgument` on `clues` |

**The distinct-token bound refuses rather than prunes**, deliberately: pruning rare tokens would
make the counts something other than an exact sum over the trained messages, and both exact
untraining and rebuilding a model from its file depend on that. When a model is full, untrain old
messages or start another model.

**No recursive parse of request input.** The tokenizer is one linear pass with nothing nested to
recurse into, so there is no depth to bound. The only recursive parse is of a model file this
crate wrote, by `serde_json`, which refuses nesting past 128 levels.

## How well it works

On a public corpus, the [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
(Almeida, Gómez Hidalgo and Yamakami, DocEng 2011; CC BY 4.0): 5,574 messages, 5,171 after
removing exact duplicates (653 spam, 4,518 ham), five-fold cross-validation, tokenizer version 1:

| cutoffs (ham, spam) | spam precision | spam recall | ham called spam | spam called ham | unsure |
|---|---|---|---|---|---|
| 0.20, 0.90 (default) | 97.61% | 87.75% | 0.31% (14) | 1.53% (10) | 15.24% |
| 0.30, 0.90 | 97.61% | 87.75% | 0.31% (14) | 2.30% (15) | 9.44% |
| 0.40, 0.80 | 94.98% | 89.89% | 0.69% (31) | 3.68% (24) | 5.84% |

At the defaults, the 718 unsure ham are shorter than the ham called ham (51 bytes and 5.6 clues on
average, against 74 bytes): short texts give few clues, and the method says so rather than guess.
Contact-form and email messages are longer than SMS. Read the table as a check that the method
works, not as a prediction for your traffic: evaluate on your own messages (below).

The corpus is not in this repository (it holds real people's messages): `scripts/corpus-eval.sh
<work-dir>` downloads it, checks a pinned SHA-256, and runs the evaluation. CI never runs it.

### Evaluating your own messages

```text
cargo run --release --example confusion -- <spam-dir> <ham-dir> [--folds 5] [--cutoffs 0.2,0.9]
    [--save <state-dir> --model contact]
```

Two directories of plain-text messages, one per file. It prints the confusion matrix (truth by
spam / unsure / ham) and the rates above; with `--save` it then trains one model on everything and
writes it into a state directory for a host to open. Your messages stay on your machine.

## Conformance

`tests/conformance.rs` runs [`ikigai-conformance`](https://github.com/ikigai-rs/ikigai-conformance)
0.8.2 over two models: zero findings across 9 endpoints, every check selected, the write doors held
to `REQUIRED` too.

## Accessibility

The scores, training outcomes and refusals this crate writes, the rules they follow, how they are tested and their known limitations are in [ACCESSIBILITY.md](ACCESSIBILITY.md). Report a barrier with an issue carrying the `accessibility` label.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
