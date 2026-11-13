# Accessibility

What `ikigai-bayes` emits that people read, the rules that output follows, how it is tested, and
how to report a barrier. It is not a conformance claim. Adapted from the
[ACCESSIBILITY.md template](https://github.com/mgifford/ACCESSIBILITY.md) (MIT).

| Field | Value |
|---|---|
| Target | WCAG 2.2 Level AA, for any page that presents this output; this repository writes text only (plain text, JSON and Turtle), no HTML or SVG |
| Status | Evaluation in progress. No conformance claim. |
| Owner | the maintainers of this repository |
| Report a barrier | [a GitHub issue](https://github.com/ikigai-rs/ikigai-bayes/issues/new?labels=accessibility) with the `accessibility` label; anything security-sensitive through [private vulnerability reporting](https://github.com/ikigai-rs/ikigai-bayes/security/advisories/new) |
| Last reviewed | 2026-11-13 (UTC) |

## What it emits that people read

- **A score** from `urn:bayes:{model}:score` (`score_text` in `src/endpoints.rs`): the class
  and the score on the first line (`spam 0.920316`), the message's digest, the model's digest,
  `clues <shown> of <combined>`, then one `<probability> <token>` line per clue, strongest first.
  Also as JSON and Turtle.
- **A training outcome** from `:train` and `:untrain`: `added …`, `moved … from … to …`,
  `unchanged …: already …` or what was removed, then the message counts.
- **Counts** from `:stats` (`model`, `messages`, `tokens`, `cutoffs`, `tokenizer`,
  `modelDigest`, one per line), and the model names from `urn:bayes:models`, one per line.
- **Refusals** (`src/endpoints.rs`, `src/lib.rs`): a face, label, `clues=` or `digest=` outside
  its form, a message past the size bound (`refused, not truncated`), a missing `content`, a
  message never trained (`NotFound`), a model configuration that does not hold.
- **`examples/confusion.rs`**: the evaluation's confusion matrix and rates, printed to a
  terminal, most rates followed by what they mean in words.

Not this repository's: the clue tokens, which are words from the scored message (prefixed by
the field they came from); the class names when a host renames its labels; the sentence each
error kind is wrapped in (ikigai-core's `Display`); and any page that shows a score to a person.

## The rules that output follows

- The class is a word, first: `spam`, `ham` or `unsure` (or the host's own label names), never a
  number alone or a color. No color is written.
- A message past the bound is refused, never cut short, and the refusal says both its size and
  the bound.
- An invalid input is named and the refusal says the form it takes (for `clues=`, an integer
  from 0 to the maximum; for `as=`, the faces served).

## How it is tested

Every named check has a result: Pass, Fail or Blocked, with evidence. A Pass covers only
its stated assertions, not overall accessibility. The passing checks below ran on 2026-11-13 (UTC)
on main commit `5b9420eb51905b3dfcc98e633e57f2a4461731fd`, before this documentation change,
using Cargo 1.99.0 and rustc 1.99.0. `cargo test` runs these tests in CI through
`.github/workflows/ci.yml`. No historical browser or assistive technology run is recorded.
This reporting format follows [gonk-starter PR 91](https://github.com/ikigai-rs/ikigai-gonk-starter/pull/91).

| Check | Result | Evidence |
|---|---|---|
| Documented resource text and structured faces | Pass | `cargo test`: doctests in `src/lib.rs` assert example score, training, untraining, stats and model-list answers, including the score's word-first line and JSON/Turtle examples. These are selected examples, not every possible answer or a usability evaluation. |
| Host-defined labels | Pass | `cargo test`: `tests/labels.rs::yes_no_labels_reach_every_face_and_the_contract` checks custom labels in the contract, training, score faces, counts and refusals. Does not assess whether a host's labels are understandable. |
| Typed input refusals and configuration errors | Pass | `cargo test`: `tests/resources.rs::bad_calls_are_refused_typed` checks error kinds and argument names for size, clues, face, label, missing input and digest errors; `a_bad_model_configuration_is_refused_at_open` checks selected configuration messages. Does not verify every refusal's full wording, size/bound values or suggested correction. |
| Training outcomes, counts and model names | Pass | `cargo test`: `tests/resources.rs` checks added/unchanged/moved text, exact untraining and stats examples, and the model list. Does not evaluate comprehension or assistive technology presentation. |
| Complete refusal wording and no-color output audit | Blocked | No complete audit of all emitted strings was run. The tests above cover selected assertions only; the output rules above remain requirements. |
| Example confusion matrix and rate readability | Blocked | No terminal usability evaluation of `examples/confusion.rs` was run. Its space-aligned matrix limitation below remains unresolved. |
| Automated accessibility checker | Blocked | No axe or other automated accessibility checker was run; no rendered host page or browser checker is configured in this repository. Rust tests do not replace this check. |
| Browser rendering, zoom, reflow and contrast | Blocked | No browser evaluation was run. This library emits text, JSON and Turtle; a host page, terminal or documentation viewer determines presentation and needs its own evaluation. |
| Manual keyboard | Blocked | No person performed a keyboard pass in a host, terminal or documentation viewer for this review. |
| Screen reader | Blocked | No screen reader session was available or used on this output for this review. |
| Screen magnifier | Blocked | No magnifier session was available or used on this output for this review. |
| Speech input | Blocked | No speech-input session was available or used on this output for this review. |
| Switch access | Blocked | No switch-access session was available or used on this output for this review. |
| Braille display | Blocked | No braille display was available or used on this output for this review. |

## Known limitations

Found by reading the source on 2026-10-31:

- The text faces are `key value` lines with code-style keys (`modelDigest`, `clues 2 of 2`) and
  six-decimal probabilities; nothing says in words what a score or a clue's probability means.
- `ham` is filter jargon for "not spam" unless the host names its labels.
- A clue token is the tokenizer's form, and some are made up rather than read (`skip:a 30` for a
  long run of characters, `links:…` for a count of links), which a reader cannot decode without
  the README.
- The example's confusion matrix lines its columns up with spaces, which assumes a fixed-width
  font and reads as a run of numbers to a screen reader.
- All text is English, with no way to choose another language.

## Contributing

Read this file before changing any output listed above: the faces, the refusals or the example.
Keep the class a word, first. In the pull request's accessibility section, give every named
check a row in a `Check | Result | Evidence` table, using Pass, Fail or Blocked. Include the
test or command and tool version, or why it could not run, and state the coverage limits.
Keep unrun checks as Blocked, including each assistive technology nobody used. A historical
run that CI does not repeat is Blocked with its date, not Pass.

## Reporting a barrier

[Open an issue](https://github.com/ikigai-rs/ikigai-bayes/issues/new?labels=accessibility) with
the `accessibility` label (the link sets it when GitHub lets you add labels; otherwise a
maintainer adds it): what you were doing, what happened, and your terminal or other tool and any
assistive technology, if you are comfortable sharing them. Anything security-sensitive goes
through [private vulnerability reporting](https://github.com/ikigai-rs/ikigai-bayes/security/advisories/new).
**We acknowledge a report within 7 days**, and promise nothing further.

## Change history

| Date | Change |
|---|---|
| 2026-10-31 | First version. |
| 2026-11-13 | Record check results, main test evidence and blocked evaluations; require the same evidence format in contributions. |
