//! Where a model's bytes live, behind a small trait so the classifier stays wasm-clean.
//!
//! [`ResourceStore`] keeps them in RESOURCES: it reads and writes a model with Source and Sink
//! requests on IRIs under a base the host chooses (`urn:file:` over an `ikigai-fs` mount, or
//! anything else that answers Source and Sink), through a kernel and capability the host hands
//! over. The classifier never opens a file itself (Hypothesis H, ledger #2384).
//!
//! [`DirStore`] is the host's state directory, handed over by the HOST (as `ikigai-layers` takes
//! `--layers-state`), never an environment variable or a hidden path under `$HOME`. It is a
//! [`ResourceStore`] over a private `ikigai-fs` mount jailed to that directory, plus the
//! directory's one-writer lock. Neither is built for `wasm32`; [`MemoryStore`] is (and is what
//! tests use).
//!
//! # The resources
//!
//! ```text
//! {base}models              the registry: one model name per line, every model ever created
//! {base}<model>.json        the model (see `Model::to_bytes`)
//! ```
//!
//! and, for a [`DirStore`], with `{base}` = `urn:file:` jailed to the directory:
//!
//! ```text
//! <state>/LOCK              held exclusively for the store's life: one writer per directory
//! <state>/models            the registry
//! <state>/<model>.json      the model
//! ```
//!
//! Every write is a Sink, so its atomicity is the resource's: `ikigai-fs` stages the bytes,
//! syncs them and renames them over the file, so a crash leaves the old file or the new one,
//! never half of one.
//!
//! **Missing or corrupt is refused by name; only a NEW model starts empty.** A model is new when
//! the registry does not list it and its resource is NotFound. A model the registry lists whose
//! resource is gone is refused, because starting it empty would silently discard everything it
//! learned. A resource present but unregistered (a registry lost, or a file copied in) is read
//! like any other, and registered. What this cannot see is a store with BOTH the model and the
//! registry entry gone: that is indistinguishable from a new one.

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
pub use resource::ResourceStore;

#[cfg(not(target_family = "wasm"))]
mod resource {
    use std::collections::BTreeSet;
    use std::sync::{Arc, Mutex};

    use futures_executor::block_on;
    use ikigai_core::{ArgRef, Capability, Error, Iri, Kernel, Representation, Request, Verb};

    use super::{Store, StoreError};
    use crate::model::MAX_MODEL_BYTES;

    /// The registry's name under the base.
    const REGISTRY: &str = "models";

    /// A store whose models are resources: `{base}<model>.json` and the registry
    /// `{base}models`, read with Source and written with Sink through a kernel the host hands
    /// over, under the capability it hands over. See the module documentation.
    ///
    /// Each request is driven to completion on a short-lived thread while the caller waits (a
    /// train blocks on its write, as it did on a file). So the kernel must answer without the
    /// caller's executor, and must not be the one that serves this [`Bayes`](crate::Bayes):
    /// give the store its own, over the space that holds the models.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use ikigai_bayes::store::{ResourceStore, Store};
    /// use ikigai_core::{Capability, Kernel};
    ///
    /// // Anything that answers Source and Sink under the base will do; here, an `ikigai-fs`
    /// // mount the host binds.
    /// # let dir = std::env::temp_dir().join(format!("bayes-doc-{}", std::process::id()));
    /// # std::fs::create_dir_all(&dir).unwrap();
    /// let kernel = Arc::new(Kernel::new(Arc::new(ikigai_fs::space(&dir))));
    /// let store = ResourceStore::open(kernel, "urn:file:", Capability::root()).unwrap();
    /// assert_eq!(store.load("contact").unwrap(), None); // new
    /// store.save("contact", b"{}").unwrap();
    /// assert_eq!(store.load("contact").unwrap().as_deref(), Some(&b"{}"[..]));
    /// # std::fs::remove_dir_all(&dir).unwrap();
    /// ```
    pub struct ResourceStore {
        kernel: Arc<Kernel>,
        base: String,
        capability: Capability,
        registry: Mutex<BTreeSet<String>>,
    }

    impl std::fmt::Debug for ResourceStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("ResourceStore")
                .field("base", &self.base)
                .finish()
        }
    }

    impl ResourceStore {
        /// Open the models under `base` (an IRI prefix such as `urn:file:` or
        /// `urn:file:bayes/`), reading the registry `{base}models` through `kernel` under
        /// `capability`. A registry that is NotFound is empty; any other failure is refused.
        pub fn open(
            kernel: Arc<Kernel>,
            base: impl Into<String>,
            capability: Capability,
        ) -> Result<ResourceStore, StoreError> {
            let store = ResourceStore {
                kernel,
                base: base.into(),
                capability,
                registry: Mutex::new(BTreeSet::new()),
            };
            let registry = store.iri(REGISTRY)?;
            let names = match store.issue(Request::new(Verb::Source, registry.clone())) {
                Ok(rep) => String::from_utf8(rep.bytes)
                    .map_err(|e| StoreError(format!("read {}: not UTF-8: {e}", registry.as_str())))?
                    .lines()
                    .map(str::trim)
                    .filter(|l| !l.is_empty())
                    .map(String::from)
                    .collect(),
                Err(Error::NotFound(_)) => BTreeSet::new(),
                Err(e) => return Err(StoreError(format!("read {}: {e}", registry.as_str()))),
            };
            *store.registry.lock().expect("registry lock") = names;
            Ok(store)
        }

        /// The base the models live under.
        pub fn base(&self) -> &str {
            &self.base
        }

        fn iri(&self, name: &str) -> Result<Iri, StoreError> {
            let iri = format!("{}{name}", self.base);
            Iri::parse(&iri).map_err(|e| StoreError(format!("`{iri}` is not an IRI: {e}")))
        }

        fn model_iri(&self, model: &str) -> Result<Iri, StoreError> {
            self.iri(&format!("{model}.json"))
        }

        /// Drive one request to completion on a thread of its own. A `Store` is synchronous and
        /// is called from INSIDE a request an executor is already driving (a train is a Sink
        /// the host's kernel serves), where a nested `block_on` panics (`EnterError`); a fresh
        /// thread has no executor entered, so the store's request runs there while the caller
        /// waits, as it waited on a file before.
        fn issue(&self, request: Request) -> ikigai_core::Result<Representation> {
            std::thread::scope(|s| {
                s.spawn(|| block_on(self.kernel.issue(request, &self.capability)))
                    .join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
            })
        }

        fn write(&self, iri: &Iri, bytes: &[u8]) -> Result<(), StoreError> {
            let request = Request::new(Verb::Sink, iri.clone())
                .with_arg("content", ArgRef::Inline(bytes.to_vec()));
            self.issue(request)
                .map(drop)
                .map_err(|e| StoreError(format!("write {}: {e}", iri.as_str())))
        }

        fn register(&self, model: &str) -> Result<(), StoreError> {
            let mut registry = self.registry.lock().expect("registry lock");
            if registry.contains(model) {
                return Ok(());
            }
            let mut next = registry.clone();
            next.insert(model.to_string());
            let text: String = next.iter().map(|m| format!("{m}\n")).collect();
            self.write(&self.iri(REGISTRY)?, text.as_bytes())?;
            *registry = next;
            Ok(())
        }
    }

    impl Store for ResourceStore {
        fn load(&self, model: &str) -> Result<Option<Vec<u8>>, StoreError> {
            let iri = self.model_iri(model)?;
            let registered = self.registry.lock().expect("registry lock").contains(model);
            let bytes = match self.issue(Request::new(Verb::Source, iri.clone())) {
                Ok(rep) => rep.bytes,
                Err(Error::NotFound(_)) => {
                    if registered {
                        return Err(StoreError(format!(
                            "model `{model}` is registered in {} but its resource {} is \
                             missing: refusing to start it empty, which would discard what it \
                             learned",
                            self.iri(REGISTRY)?.as_str(),
                            iri.as_str()
                        )));
                    }
                    return Ok(None);
                }
                Err(e) => return Err(StoreError(format!("read {}: {e}", iri.as_str()))),
            };
            // The resource bounds what it answers (a kernel's run byte budget, 64 MiB by
            // default); this is the model's own bound.
            if bytes.len() > MAX_MODEL_BYTES {
                return Err(StoreError(format!(
                    "{} is past the bound of {MAX_MODEL_BYTES} bytes",
                    iri.as_str()
                )));
            }
            if !registered {
                self.register(model)?;
            }
            Ok(Some(bytes))
        }

        fn save(&self, model: &str, bytes: &[u8]) -> Result<(), StoreError> {
            self.write(&self.model_iri(model)?, bytes)?;
            self.register(model)
        }
    }
}

#[cfg(not(target_family = "wasm"))]
mod dir {
    use std::fs::{File, OpenOptions, TryLockError};
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use ikigai_core::{Capability, Kernel};

    use super::{ResourceStore, Store, StoreError};

    /// The lock file's name.
    const LOCK_FILE: &str = "LOCK";

    /// A state directory on disk: a [`ResourceStore`] over a private `ikigai-fs` mount
    /// (`urn:file:`) jailed to the directory, and the directory's one-writer lock. See the
    /// module documentation.
    pub struct DirStore {
        dir: PathBuf,
        /// Held, never read: the OS releases the lock when the file closes, so a crashed host
        /// never leaves the directory locked.
        _lock: File,
        models: ResourceStore,
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
            let lock = claim(&dir)?;
            // The kernel is private and binds nothing but the jail, so `root` reaches exactly
            // the directory the host handed over: the jail confines it, not the capability.
            // (A scoped `urn:cap:fs:*:<dir>` grant would need the path as UTF-8 text.)
            let kernel = Kernel::new(Arc::new(ikigai_fs::space(&dir)));
            let models = ResourceStore::open(Arc::new(kernel), "urn:file:", Capability::root())
                .map_err(|e| in_dir(&dir, e))?;
            Ok(DirStore {
                dir,
                _lock: lock,
                models,
            })
        }

        /// The directory.
        pub fn dir(&self) -> &Path {
            &self.dir
        }
    }

    impl Store for DirStore {
        fn load(&self, model: &str) -> Result<Option<Vec<u8>>, StoreError> {
            self.models.load(model).map_err(|e| in_dir(&self.dir, e))
        }

        fn save(&self, model: &str, bytes: &[u8]) -> Result<(), StoreError> {
            self.models
                .save(model, bytes)
                .map_err(|e| in_dir(&self.dir, e))
        }
    }

    /// A resource error, saying which directory `urn:file:` is jailed to (the error itself
    /// names only the IRI, never a host path).
    fn in_dir(dir: &Path, e: StoreError) -> StoreError {
        StoreError(format!("{e} (urn:file: is {})", dir.display()))
    }

    /// Create the directory if needed, then take `<dir>/LOCK` exclusively and write this
    /// process's id into it. The only file access this crate makes itself: `ikigai-fs` has no
    /// lock, and the one-writer rule is what keeps two processes from each training a copy of
    /// a model and losing the other's work.
    fn claim(dir: &Path) -> Result<File, StoreError> {
        // ambient: fs: creates the host-handed state directory, so the ikigai-fs jail has a root to open
        #[allow(clippy::disallowed_methods)]
        let made = std::fs::create_dir_all(dir);
        made.map_err(|e| StoreError(format!("create {}: {e}", dir.display())))?;
        let path = dir.join(LOCK_FILE);
        // Never truncated on open: until the lock is ours, the pid in it is the holder's.
        // ambient: fs: the host-handed state directory's one-writer lock file; ikigai-fs has no lock resource
        #[allow(clippy::disallowed_methods)]
        let opened = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path);
        let mut file = opened.map_err(|e| StoreError(format!("open {}: {e}", path.display())))?;
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
