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
| Last reviewed | 2026-10-31 |

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

- The doctests in `src/lib.rs` hold every answer's exact text, the score's first line included.
- `tests/labels.rs` (`yes_no_labels_reach_every_face_and_the_contract`) checks a host's own
  label names reach every face; `tests/resources.rs` (`bad_calls_are_refused_typed`) checks each
  bad input is refused naming that input.
- **No automated accessibility checker, browser or assistive technology has been used on this
  output.**

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
Keep the class a word, first. Say in the pull request what was tested, with what, and what was
not.

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
