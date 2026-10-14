//! Where a model's bytes live, behind a small trait so the classifier stays wasm-clean.
//!
//! [`DirStore`] is the host's: a state directory the HOST hands over (as `ikigai-layers` takes
//! `--layers-state`), never an environment variable or a hidden path under `$HOME`. It is not
//! built for `wasm32`, where there is no filesystem; [`MemoryStore`] is (and is what tests use).
//!
//! # The directory
//!
//! ```text
//! <state>/LOCK              held exclusively for the store's life: one writer per directory
//! <state>/models            the registry: one model name per line, every model ever created
//! <state>/<model>.json      the model (see `Model::to_bytes`)
//! ```
//!
//! Every write is atomic: the bytes go to `<file>.tmp`, are synced, and are renamed over the
//! file, so a crash leaves the old file or the new one, never half of one.
//!
//! **Missing or corrupt is refused by name; only a NEW model starts empty.** A model is new when
//! the registry does not list it and it has no file. A model the registry lists whose file is
//! gone is refused, because starting it empty would silently discard everything it learned. A
//! file present but unregistered (a registry lost, or a file copied in) is read like any other,
//! and registered. What this cannot see is a state directory with BOTH the file and the registry
//! entry gone: that is indistinguishable from a new directory.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Mutex;

/// A store's failure, naming what failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoreError(pub String);

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for StoreError {}

/// Where model bytes are kept.
pub trait Store: Send + Sync {
    /// A model's bytes, or `None` when the model is NEW here (never created). A model this store
    /// knows was created but cannot read is an error, never `None`.
    fn load(&self, model: &str) -> Result<Option<Vec<u8>>, StoreError>;

    /// Replace a model's bytes atomically, creating (and registering) it if it is new.
    fn save(&self, model: &str, bytes: &[u8]) -> Result<(), StoreError>;
}

/// A store in memory: nothing survives the process. For tests and for a host with no disk.
#[derive(Default)]
pub struct MemoryStore {
    models: Mutex<BTreeMap<String, Vec<u8>>>,
}

impl MemoryStore {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Store for MemoryStore {
    fn load(&self, model: &str) -> Result<Option<Vec<u8>>, StoreError> {
        Ok(self.models.lock().expect("store lock").get(model).cloned())
    }

    fn save(&self, model: &str, bytes: &[u8]) -> Result<(), StoreError> {
        self.models
            .lock()
            .expect("store lock")
            .insert(model.to_string(), bytes.to_vec());
        Ok(())
    }
}

#[cfg(not(target_family = "wasm"))]
pub use dir::DirStore;

#[cfg(not(target_family = "wasm"))]
mod dir {
    use std::collections::BTreeSet;
    use std::fs::{File, OpenOptions, TryLockError};
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;

    use super::{Store, StoreError};
    use crate::model::MAX_MODEL_BYTES;

    /// The lock file's name.
    const LOCK_FILE: &str = "LOCK";
    /// The registry's name.
    const REGISTRY: &str = "models";

    /// A state directory on disk. See the module documentation.
    pub struct DirStore {
        dir: PathBuf,
        /// Held, never read: the OS releases the lock when the file closes, so a crashed host
        /// never leaves the directory locked.
        _lock: File,
        registry: Mutex<BTreeSet<String>>,
    }

    impl std::fmt::Debug for DirStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("DirStore").field("dir", &self.dir).finish()
        }
    }

    impl DirStore {
        /// Open a state directory, creating it if needed and taking its lock. Refused when
        /// another process holds it: one writer per directory, and so one per model.
        pub fn open(dir: impl AsRef<Path>) -> Result<DirStore, StoreError> {
            let dir = dir.as_ref().to_path_buf();
            std::fs::create_dir_all(&dir)
                .map_err(|e| StoreError(format!("create {}: {e}", dir.display())))?;
            let lock = lock(&dir)?;
            let path = dir.join(REGISTRY);
            let registry = match std::fs::read_to_string(&path) {
                Ok(text) => text
                    .lines()
                    .map(str::trim)
                    .filter(|l| !l.is_empty())
                    .map(String::from)
                    .collect(),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => BTreeSet::new(),
                Err(e) => return Err(StoreError(format!("read {}: {e}", path.display()))),
            };
            Ok(DirStore {
                dir,
                _lock: lock,
                registry: Mutex::new(registry),
            })
        }

        /// The directory.
        pub fn dir(&self) -> &Path {
            &self.dir
        }

        fn model_path(&self, model: &str) -> PathBuf {
            self.dir.join(format!("{model}.json"))
        }
    }

    impl Store for DirStore {
        fn load(&self, model: &str) -> Result<Option<Vec<u8>>, StoreError> {
            let path = self.model_path(model);
            let registered = self.registry.lock().expect("registry lock").contains(model);
            let mut file = match File::open(&path) {
                Ok(file) => file,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    if registered {
                        return Err(StoreError(format!(
                            "model `{model}` is registered in {} but its file {} is missing: \
                             refusing to start it empty, which would discard what it learned",
                            self.dir.join(REGISTRY).display(),
                            path.display()
                        )));
                    }
                    return Ok(None);
                }
                Err(e) => return Err(StoreError(format!("open {}: {e}", path.display()))),
            };
            // Read one byte past the bound, so an oversized file is refused without reading it
            // whole.
            let mut bytes = Vec::new();
            (&mut file)
                .take(MAX_MODEL_BYTES as u64 + 1)
                .read_to_end(&mut bytes)
                .map_err(|e| StoreError(format!("read {}: {e}", path.display())))?;
            if bytes.len() > MAX_MODEL_BYTES {
                return Err(StoreError(format!(
                    "{} is past the bound of {MAX_MODEL_BYTES} bytes",
                    path.display()
                )));
            }
            if !registered {
                self.register(model)?;
            }
            Ok(Some(bytes))
        }

        fn save(&self, model: &str, bytes: &[u8]) -> Result<(), StoreError> {
            atomic_write(&self.model_path(model), bytes)?;
            self.register(model)
        }
    }

    impl DirStore {
        fn register(&self, model: &str) -> Result<(), StoreError> {
            let mut registry = self.registry.lock().expect("registry lock");
            if registry.contains(model) {
                return Ok(());
            }
            let mut next = registry.clone();
            next.insert(model.to_string());
            let text: String = next.iter().map(|m| format!("{m}\n")).collect();
            atomic_write(&self.dir.join(REGISTRY), text.as_bytes())?;
            *registry = next;
            Ok(())
        }
    }

    /// Write to `<path>.tmp`, sync it, and rename it over `path`.
    fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), StoreError> {
        let mut tmp = path.as_os_str().to_owned();
        tmp.push(".tmp");
        let tmp = PathBuf::from(tmp);
        let write = || -> std::io::Result<()> {
            let mut file = File::create(&tmp)?;
            file.write_all(bytes)?;
            file.sync_all()?;
            std::fs::rename(&tmp, path)
        };
        write().map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            StoreError(format!("write {}: {e}", path.display()))
        })
    }

    /// Take `<dir>/LOCK` exclusively and write this process's id into it.
    fn lock(dir: &Path) -> Result<File, StoreError> {
        let path = dir.join(LOCK_FILE);
        // Never truncated on open: until the lock is ours, the pid in it is the holder's.
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(|e| StoreError(format!("open {}: {e}", path.display())))?;
        match file.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => {
                let mut held = String::new();
                let holder = match file.read_to_string(&mut held) {
                    Ok(_) if !held.trim().is_empty() => format!("pid {}", held.trim()),
                    _ => "a pid it has not written".to_string(),
                };
                return Err(StoreError(format!(
                    "state directory {} is held by another writer ({holder}): one writer per \
                     directory, so that every model has one",
                    dir.display()
                )));
            }
            Err(TryLockError::Error(e)) => {
                return Err(StoreError(format!("lock {}: {e}", path.display())));
            }
        }
        file.set_len(0)
            .and_then(|()| file.seek(SeekFrom::Start(0)))
            .and_then(|_| writeln!(file, "{}", std::process::id()))
            .and_then(|()| file.flush())
            .map_err(|e| StoreError(format!("write {}: {e}", path.display())))?;
        Ok(file)
    }
}
