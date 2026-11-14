//! A model's state, sealed at rest.
//!
//! A model file holds the token set of every message it was taught (that is what makes
//! untraining by digest exact), so it is as sensitive as a bag of words of those messages. A
//! [`SealedStore`] keeps it as [age](https://age-encryption.org/v1) ciphertext (X25519), sealed
//! to the host's own recipient and opened with the host's own identity, as `ikigai-layers` seals
//! its state: the files, a backup of them or a disk image are unreadable without the key.
//!
//! - Each model (`<model>.json`) is ONE age file, binary, with its own random file key wrapped
//!   to the recipient. The name keeps its suffix: what is in a file says what it is (a sealed
//!   file begins `age-encryption.org/v1`).
//! - The registry (`models`) stays plaintext: it holds only the model names the host chose.
//! - What a model scores, its counts and its trained-state digest (`modelDigest`) are those of
//!   the bytes inside, so sealing changes none of them.
//!
//! The KEY is an age identity file, as `age-keygen -o` writes it, handed over by the HOST as a
//! PATH ([`StateKey::load`]) or as its text ([`StateKey::parse`]), so a deployment can hand it
//! over from a credential store. It may hold several identities: models are sealed to the
//! FIRST (the current one) and opened with any. That is rotation: put a new identity first, keep
//! the old one after it, open once with [`Migration::Reseal`], then drop the old line.
//!
//! A model that is plaintext (written before its state was sealed) is refused by name until the
//! host opens once with [`Migration::Reseal`], which seals it in place. A model sealed to a key
//! the identity file does not hold is refused, never started empty.
//!
//! What this does NOT protect against: the running host, which holds the counts in memory to
//! score, and anyone who holds the identity file.
//!
//! ```
//! use std::sync::Arc;
//! use ikigai_bayes::at_rest::{Migration, SealedStore, StateKey};
//! use ikigai_bayes::store::{MemoryStore, Store};
//! use ikigai_encrypt::age::{ExposeSecret, Identity};
//!
//! let key = StateKey::parse(Identity::generate().to_string().expose_secret()).unwrap();
//! let disk = Arc::new(MemoryStore::new());
//! let store = SealedStore::new(disk.clone(), &key);
//! store.save("contact", b"{\"tokens\":[\"pills\"]}").unwrap();
//! let at_rest = disk.load("contact").unwrap().unwrap();
//! assert!(at_rest.starts_with(b"age-encryption.org/v1\n"));
//! assert_eq!(store.load("contact").unwrap().unwrap(), b"{\"tokens\":[\"pills\"]}");
//! assert_eq!(Migration::default(), Migration::None);
//! ```

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ikigai_encrypt::age::{self, Identity, Recipient};

use crate::store::{Store, StoreError};

/// The first line of every age file (age v1, binary).
pub const MAGIC: &[u8] = b"age-encryption.org/v1";

/// What [`Bayes::open_sealed_with`](crate::Bayes::open_sealed_with) does with a model that is
/// not sealed to the current key.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Migration {
    /// Refuse a model that is plaintext, naming it and this option. Models sealed to any
    /// identity in the key open; nothing is rewritten.
    #[default]
    None,
    /// Before anything is read, seal every plaintext model in place (the one-time migration
    /// from a host that wrote plaintext), and seal again, to the current identity, every model
    /// sealed only to an older one (rotation): every model the store has registered and every
    /// model offered. Each is one atomic write, so a model is whole in its old form or its new
    /// one, and an interrupted run is resumed by running it again.
    Reseal,
}

/// The host's key: age identities, read once.
///
/// Cheap to clone. Its `Debug` names where it came from and the current recipient, never a
/// secret.
#[derive(Clone)]
pub struct StateKey(Arc<KeyInner>);

struct KeyInner {
    from: Option<PathBuf>,
    identities: Vec<Identity>,
    recipient: Recipient,
}

impl fmt::Debug for StateKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StateKey")
            .field("path", &self.0.from)
            .field("recipient", &self.recipient())
            .finish()
    }
}

impl StateKey {
    /// Read the age identity file at `path` (see [`StateKey::parse`]). Refused, naming the path
    /// and never echoing a line of it: a file that does not exist or cannot be read, and what
    /// `parse` refuses. On unix, also refused before a byte of it is read, with the command
    /// that fixes it: a file group or others can reach (but for a root-owned file readable by
    /// group root, as systemd's credential store writes it), and a directory holding it that
    /// group or others can write without the sticky bit.
    pub fn load(path: impl AsRef<Path>) -> Result<StateKey, StoreError> {
        let path = path.as_ref();
        let refuse = |why: String| {
            StoreError(format!(
                "the state identity {}: {why}; the models are sealed to it and nothing opens \
                 without it",
                path.display()
            ))
        };
        // ambient: fs: the identity file whose path the host hands over, read once at open
        #[allow(clippy::disallowed_methods)]
        let opened = std::fs::File::open(path);
        let mut file = opened.map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => refuse(
                "does not exist (an age identity file, as `age-keygen -o` writes one)".to_string(),
            ),
            _ => refuse(format!("cannot be read ({e})")),
        })?;
        #[cfg(unix)]
        check_key_access(path, &file).map_err(refuse)?;
        let mut text = String::new();
        std::io::Read::read_to_string(&mut file, &mut text)
            .map_err(|e| refuse(format!("cannot be read ({e})")))?;
        let identities = identities(&text).map_err(refuse)?;
        Ok(StateKey::of(Some(path.to_path_buf()), identities))
    }

    /// The text of an age identity file: one `AGE-SECRET-KEY-1…` per line, blank lines and `#`
    /// comments skipped, the first the current identity. Refused, never echoing a line of it:
    /// text that holds no identity, and a line that is not an X25519 age identity (named by
    /// its number).
    pub fn parse(text: &str) -> Result<StateKey, StoreError> {
        let identities = identities(text).map_err(|why| {
            StoreError(format!(
                "the state identity: {why}; the models are sealed to it"
            ))
        })?;
        Ok(StateKey::of(None, identities))
    }

    fn of(from: Option<PathBuf>, identities: Vec<Identity>) -> StateKey {
        // `parse_identities` refuses an empty list, so there is a first.
        let recipient = identities[0].to_public();
        StateKey(Arc::new(KeyInner {
            from,
            identities,
            recipient,
        }))
    }

    /// The current recipient (`age1…`): what every model is sealed to now.
    pub fn recipient(&self) -> String {
        self.0.recipient.to_string()
    }
}

fn identities(text: &str) -> Result<Vec<Identity>, String> {
    age::parse_identities(text).map_err(|e| match e {
        age::Error::BadKey { line, detail } => {
            format!("line {line} is not an X25519 age identity ({detail})")
        }
        age::Error::NoKeys => {
            "holds no age identity (expected a line AGE-SECRET-KEY-1…)".to_string()
        }
        other => other.to_string(),
    })
}

/// A [`Store`] that seals every model it saves to a [`StateKey`] and opens every model it
/// loads with it, over any other store (a [`DirStore`](crate::store::DirStore), a
/// [`ResourceStore`](crate::store::ResourceStore)). See the module documentation.
pub struct SealedStore {
    inner: Arc<dyn Store>,
    key: StateKey,
}

impl fmt::Debug for SealedStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SealedStore")
            .field("recipient", &self.key.recipient())
            .finish()
    }
}

impl SealedStore {
    /// Seal the models `inner` keeps with `key`. A plaintext model is refused at load.
    pub fn new(inner: Arc<dyn Store>, key: &StateKey) -> SealedStore {
        SealedStore {
            inner,
            key: key.clone(),
        }
    }

    /// Seal in place, to the current identity, every model that is plaintext or sealed only to
    /// an older identity: every model the store has registered, and `also` (the models a host
    /// offers, in case one is present but unregistered). Answers the models it sealed, in order.
    /// A model sealed to no identity in the key, or one that does not open whole, is refused by
    /// name, and nothing after it is sealed; running it again resumes.
    pub fn reseal<'a>(
        &self,
        also: impl IntoIterator<Item = &'a str>,
    ) -> Result<Vec<String>, StoreError> {
        let mut models = self.inner.registered()?;
        models.extend(also.into_iter().map(String::from));
        models.sort();
        models.dedup();
        let mut sealed = Vec::new();
        for model in models {
            let Some(bytes) = self.inner.load(&model)? else {
                continue;
            };
            if self.is_current(&bytes) {
                continue;
            }
            let plain = if is_plaintext(&bytes) {
                bytes
            } else {
                self.open(&model, &bytes)?
            };
            self.inner.save(&model, &self.seal(&model, &plain)?)?;
            sealed.push(model);
        }
        Ok(sealed)
    }

    fn seal(&self, model: &str, plain: &[u8]) -> Result<Vec<u8>, StoreError> {
        age::seal(plain, std::slice::from_ref(&self.key.0.recipient))
            .map_err(|e| StoreError(format!("model `{model}`: seal: {e}")))
    }

    fn open(&self, model: &str, bytes: &[u8]) -> Result<Vec<u8>, StoreError> {
        if is_plaintext(bytes) {
            return Err(StoreError(format!(
                "model `{model}` is not sealed (plaintext from before its state was sealed at \
                 rest?): open once with the migration option (`Migration::Reseal`) to seal it \
                 in place"
            )));
        }
        age::open(bytes, &self.key.0.identities).map_err(|e| match e {
            age::Error::NoMatchingKey => StoreError(format!(
                "model `{model}` is sealed to a key the state identity does not hold: the \
                 identity lacks the key it was sealed to"
            )),
            other => StoreError(format!("model `{model}` does not open whole ({other})")),
        })
    }

    /// Whether `bytes` are sealed and the CURRENT identity opens them.
    fn is_current(&self, bytes: &[u8]) -> bool {
        bytes.starts_with(MAGIC) && age::open(bytes, &self.key.0.identities[..1]).is_ok()
    }
}

impl Store for SealedStore {
    fn load(&self, model: &str) -> Result<Option<Vec<u8>>, StoreError> {
        match self.inner.load(model)? {
            Some(bytes) => self.open(model, &bytes).map(Some),
            None => Ok(None),
        }
    }

    fn save(&self, model: &str, bytes: &[u8]) -> Result<(), StoreError> {
        self.inner.save(model, &self.seal(model, bytes)?)
    }

    fn registered(&self) -> Result<Vec<String>, StoreError> {
        self.inner.registered()
    }
}

/// Whether a model's first bytes say it is plaintext: non-empty, and not (a prefix of) an age
/// file. A write torn inside an age header is a prefix of one, so it is not plaintext: it is
/// refused as not opening whole.
fn is_plaintext(bytes: &[u8]) -> bool {
    let n = bytes.len().min(MAGIC.len());
    n > 0 && bytes[..n] != MAGIC[..n]
}

/// Refuse the opened identity file at `path` if another account could read it, or replace it
/// through its directory.
#[cfg(unix)]
fn check_key_access(path: &Path, file: &std::fs::File) -> Result<(), String> {
    use std::os::unix::fs::MetadataExt;

    let unchecked = |e: std::io::Error| format!("cannot be checked ({e})");
    let meta = file.metadata().map_err(unchecked)?;
    if meta.is_dir() {
        // Reading it fails, and says so.
        return Ok(());
    }
    let mode = meta.mode() & 0o7777;
    // systemd's credential store leaves a root-owned file whose group bits are the mask of the
    // ACL naming the service.
    let group_read_ok = meta.uid() == 0 && meta.gid() == 0 && mode & 0o020 == 0;
    let closed = if group_read_ok { 0o007 } else { 0o077 };
    if mode & closed != 0 {
        return Err(format!(
            "has mode {mode:04o}, which another account could read or write: run `chmod 600 {}`",
            shell_word(path)
        ));
    }
    let dir = match path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir.to_path_buf(),
        _ => PathBuf::from("."),
    };
    // ambient: fs: the mode of the directory holding the host-handed identity file
    #[allow(clippy::disallowed_methods)]
    let dir_meta = std::fs::metadata(&dir);
    let dir_mode = dir_meta.map_err(unchecked)?.mode() & 0o7777;
    if dir_mode & 0o022 != 0 && dir_mode & 0o1000 == 0 {
        return Err(format!(
            "is in directory {}, mode {dir_mode:04o}, which another account could write: run \
             `chmod go-w {}`",
            dir.display(),
            shell_word(&dir)
        ));
    }
    Ok(())
}

/// `path` as one shell word: as it is when that is safe to paste, else single-quoted.
#[cfg(unix)]
fn shell_word(path: &Path) -> String {
    let text = path.display().to_string();
    if text
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b"/._-+=:@,".contains(&b))
    {
        text
    } else {
        format!("'{}'", text.replace('\'', r"'\''"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_torn_sealed_file_is_not_plaintext() {
        assert!(!is_plaintext(b""));
        assert!(!is_plaintext(b"age-enc"));
        assert!(!is_plaintext(b"age-encryption.org/v1\n-> X25519"));
        assert!(is_plaintext(b"{\"format\":\"ikigai-bayes/model\"}"));
    }

    /// The store's bound on a model resource leaves room for sealing a model at its own
    /// bound: an age header and 16 bytes per 64 KiB chunk.
    #[test]
    fn sealing_adds_less_than_the_store_allows() {
        let key = StateKey::parse(ikigai_encrypt::age::ExposeSecret::expose_secret(
            &Identity::generate().to_string(),
        ))
        .unwrap();
        let plain = vec![b'x'; 1024 * 1024];
        let sealed = age::seal(&plain, std::slice::from_ref(&key.0.recipient)).unwrap();
        assert!(sealed.len() - plain.len() <= plain.len() / 4096 + 4096);
    }
}
