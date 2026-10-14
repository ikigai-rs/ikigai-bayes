//! Evaluate the classifier on two directories of plain-text messages, one message per file:
//!
//! ```text
//! cargo run --release --example confusion -- <spam-dir> <ham-dir> [--folds N]
//!     [--cutoffs HAM,SPAM] [--save <state-dir> --model <name>]
//! ```
//!
//! It runs N-fold cross-validation (default 5): the messages are deduplicated by digest, sorted
//! by digest (so the split is deterministic and independent of file names), and message `i` is
//! held out in fold `i mod N`. Each fold trains a fresh model on the other folds and scores the
//! held-out messages. It prints the confusion matrix (truth by class, with the unsure column),
//! spam precision and recall, the ham false-positive rate and the unsure rate, at the default
//! cutoffs or the ones given.
//!
//! With `--save`, it then trains one model on EVERY message and writes it into a state
//! directory as model `<name>`, ready for a host to open with `Bayes::open`. It refuses to
//! overwrite a model that already exists there.
//!
//! A file that is not UTF-8 is read lossily; a message past `MAX_MESSAGE_BYTES` or the token
//! bound is skipped and counted, never truncated. Nothing here touches the network.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ikigai_bayes::classify::{Class, Settings};
use ikigai_bayes::model::{Label, Model};
use ikigai_bayes::store::{DirStore, Store};
use ikigai_bayes::{is_model_name, Message, MAX_MESSAGE_BYTES};
use ikigai_core::ContentId;

struct Args {
    spam: PathBuf,
    ham: PathBuf,
    folds: usize,
    settings: Settings,
    save: Option<(PathBuf, String)>,
}

fn usage() -> String {
    "usage: confusion <spam-dir> <ham-dir> [--folds N] [--cutoffs HAM,SPAM] \
     [--save <state-dir> --model <name>]"
        .to_string()
}

fn parse_args() -> Result<Args, String> {
    let mut positional = Vec::new();
    let mut folds = 5;
    let mut settings = Settings::default();
    let mut save_dir = None;
    let mut model = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = |flag: &str| args.next().ok_or_else(|| format!("{flag} needs a value"));
        match arg.as_str() {
            "--folds" => {
                folds = value("--folds")?
                    .parse()
                    .ok()
                    .filter(|n| *n >= 2)
                    .ok_or("--folds must be an integer of at least 2")?
            }
            "--cutoffs" => {
                let v = value("--cutoffs")?;
                let (ham, spam) = v.split_once(',').ok_or("--cutoffs is HAM,SPAM")?;
                let parse = |s: &str| s.trim().parse::<f64>().map_err(|e| e.to_string());
                settings = settings.with_cutoffs(parse(ham)?, parse(spam)?)?;
            }
            "--save" => save_dir = Some(PathBuf::from(value("--save")?)),
            "--model" => model = Some(value("--model")?),
            "-h" | "--help" => return Err(usage()),
            _ if arg.starts_with('-') => return Err(format!("unknown flag {arg}\n{}", usage())),
            _ => positional.push(PathBuf::from(arg)),
        }
    }
    let [spam, ham]: [PathBuf; 2] = positional.try_into().map_err(|_| usage())?;
    let save = match (save_dir, model) {
        (Some(dir), Some(name)) if is_model_name(&name) => Some((dir, name)),
        (Some(_), Some(name)) => return Err(format!("`{name}` is not a model name")),
        (None, None) => None,
        _ => return Err("--save and --model go together".to_string()),
    };
    Ok(Args {
        spam,
        ham,
        folds,
        settings,
        save,
    })
}

/// One usable message.
struct Sample {
    label: Label,
    tokens: BTreeSet<String>,
}

#[derive(Default)]
struct Intake {
    files: usize,
    skipped_size: usize,
    skipped_tokens: usize,
    duplicates: usize,
    conflicting: usize,
}

fn read_dir(
    dir: &Path,
    label: Label,
    samples: &mut BTreeMap<ContentId, Sample>,
    conflicting: &mut BTreeSet<ContentId>,
    intake: &mut Intake,
) -> Result<(), String> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| {
            p.is_file()
                && !p
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with('.'))
        })
        .collect();
    paths.sort();
    for path in paths {
        intake.files += 1;
        let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let message = Message::new(String::from_utf8_lossy(&bytes).into_owned());
        if message.len() > MAX_MESSAGE_BYTES {
            intake.skipped_size += 1;
            continue;
        }
        let Ok(tokens) = message.tokens() else {
            intake.skipped_tokens += 1;
            continue;
        };
        let digest = message.digest();
        if conflicting.contains(&digest) {
            intake.conflicting += 1;
            continue;
        }
        match samples.get(&digest).map(|s| s.label) {
            Some(existing) if existing == label => intake.duplicates += 1,
            Some(_) => {
                // The same text under both labels teaches nothing; drop it from both.
                intake.conflicting += 1;
                samples.remove(&digest);
                conflicting.insert(digest);
            }
            None => {
                samples.insert(digest, Sample { label, tokens });
            }
        }
    }
    Ok(())
}

/// Counts by (truth, class).
#[derive(Default)]
struct Matrix {
    cells: BTreeMap<(Label, &'static str), usize>,
}

impl Matrix {
    fn add(&mut self, truth: Label, class: Class) {
        *self.cells.entry((truth, class.as_str())).or_default() += 1;
    }

    fn get(&self, truth: Label, class: &str) -> usize {
        self.cells.get(&(truth, class)).copied().unwrap_or(0)
    }
}

fn pct(n: usize, d: usize) -> String {
    if d == 0 {
        "n/a".to_string()
    } else {
        format!("{:.2}%", 100.0 * n as f64 / d as f64)
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let mut samples = BTreeMap::new();
    let mut conflicting = BTreeSet::new();
    let mut intake = Intake::default();
    read_dir(
        &args.spam,
        Label::Spam,
        &mut samples,
        &mut conflicting,
        &mut intake,
    )?;
    read_dir(
        &args.ham,
        Label::Ham,
        &mut samples,
        &mut conflicting,
        &mut intake,
    )?;
    let samples: Vec<(ContentId, Sample)> = samples.into_iter().collect();
    if samples.len() < args.folds {
        return Err(format!(
            "{} usable messages, fewer than {} folds",
            samples.len(),
            args.folds
        ));
    }

    let mut matrix = Matrix::default();
    for fold in 0..args.folds {
        let mut model = Model::new("eval");
        for (i, (digest, sample)) in samples.iter().enumerate() {
            if i % args.folds != fold {
                model
                    .train(*digest, sample.label, &sample.tokens)
                    .map_err(|e| e.to_string())?;
            }
        }
        for (i, (_, sample)) in samples.iter().enumerate() {
            if i % args.folds == fold {
                matrix.add(
                    sample.label,
                    model.score(&sample.tokens, &args.settings).class,
                );
            }
        }
    }

    let s = &args.settings;
    let spam_total: usize = ["spam", "unsure", "ham"]
        .iter()
        .map(|c| matrix.get(Label::Spam, c))
        .sum();
    let ham_total: usize = ["spam", "unsure", "ham"]
        .iter()
        .map(|c| matrix.get(Label::Ham, c))
        .sum();
    let total = spam_total + ham_total;
    let called_spam = matrix.get(Label::Spam, "spam") + matrix.get(Label::Ham, "spam");
    let unsure = matrix.get(Label::Spam, "unsure") + matrix.get(Label::Ham, "unsure");
    println!(
        "files {} (skipped: {} over {MAX_MESSAGE_BYTES} bytes, {} over the token bound; {} duplicates, {} conflicting)",
        intake.files, intake.skipped_size, intake.skipped_tokens, intake.duplicates, intake.conflicting
    );
    println!(
        "messages {total} (spam {spam_total}, ham {ham_total}), {} folds, cutoffs ham < {} spam >= {}, tokenizer {}",
        args.folds,
        s.ham_cutoff,
        s.spam_cutoff,
        ikigai_bayes::tokenize::TOKENIZER_VERSION
    );
    println!();
    println!(
        "{:>12} {:>8} {:>8} {:>8}",
        "truth \\ class", "spam", "unsure", "ham"
    );
    for (truth, name) in [(Label::Spam, "spam"), (Label::Ham, "ham")] {
        println!(
            "{name:>12} {:>8} {:>8} {:>8}",
            matrix.get(truth, "spam"),
            matrix.get(truth, "unsure"),
            matrix.get(truth, "ham")
        );
    }
    println!();
    println!(
        "spam precision {}  (of messages called spam, how many were)",
        pct(matrix.get(Label::Spam, "spam"), called_spam)
    );
    println!(
        "spam recall    {}  (of spam, how much was called spam)",
        pct(matrix.get(Label::Spam, "spam"), spam_total)
    );
    println!(
        "ham called spam {}  (the costly mistake: a real message quarantined)",
        pct(matrix.get(Label::Ham, "spam"), ham_total)
    );
    println!(
        "spam called ham {}  (a missed spam)",
        pct(matrix.get(Label::Spam, "ham"), spam_total)
    );
    println!("unsure rate    {}", pct(unsure, total));

    if let Some((dir, name)) = args.save {
        let store = DirStore::open(&dir).map_err(|e| e.to_string())?;
        if store.load(&name).map_err(|e| e.to_string())?.is_some() {
            return Err(format!(
                "model `{name}` already exists in {}: refusing to overwrite it",
                dir.display()
            ));
        }
        let mut model = Model::new(&name);
        for (digest, sample) in &samples {
            model
                .train(*digest, sample.label, &sample.tokens)
                .map_err(|e| e.to_string())?;
        }
        store
            .save(&name, &model.to_bytes())
            .map_err(|e| e.to_string())?;
        println!();
        println!(
            "saved model `{name}` ({} spam, {} ham, {} tokens) into {}",
            model.nspam(),
            model.nham(),
            model.distinct_tokens(),
            dir.display()
        );
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
