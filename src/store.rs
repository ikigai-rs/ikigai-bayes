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
//! <state>/PENDING           a first creation awaiting recovery (only during a transaction)
//! <state>/<model>.json      the model
//! ```
//!
//! Every write is a Sink, so its atomicity is the resource's: `ikigai-fs` stages the bytes,
//! syncs them and renames them over the file, so a crash leaves the old file or the new one,
//! never half of one. For a first `DirStore` save, a journal holds the model bytes before the
//! registry is written. The registry write commits the creation: recovery discards the journal
//! if the name is absent, or completes the model write if the name is present. A failed save
//! after that commit can therefore finish on the next operation or open. Directory entries are
//! synced on Unix. A generic `ResourceStore` has no such cross-resource transaction; its Sink
//! provider must supply the needed guarantees when used directly.
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

    /// Every model this store knows was created, whether or not a host offers it now: what
    /// [`SealedStore::reseal`](crate::at_rest::SealedStore::reseal) walks. A store that keeps no
    /// such list answers none, the default.
    fn registered(&self) -> Result<Vec<String>, StoreError> {
        Ok(Vec::new())
    }
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

    fn registered(&self) -> Result<Vec<String>, StoreError> {
        Ok(self
            .models
            .lock()
            .expect("store lock")
            .keys()
            .cloned()
            .collect())
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

    /// The largest model resource read, in bytes: the model's own bound, and room for sealing
    /// it at rest ([`crate::at_rest`]), which adds an age header (a few hundred bytes) and 16
    /// bytes per 64 KiB chunk. The model inside is held to [`MAX_MODEL_BYTES`] when it is read.
    const MAX_STORED_BYTES: usize = MAX_MODEL_BYTES + MAX_MODEL_BYTES / 4096 + 4096;

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

        pub(super) fn iri(&self, name: &str) -> Result<Iri, StoreError> {
            let iri = format!("{}{name}", self.base);
            Iri::parse(&iri).map_err(|e| StoreError(format!("`{iri}` is not an IRI: {e}")))
        }

        pub(super) fn model_iri(&self, model: &str) -> Result<Iri, StoreError> {
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

        /// Sink `bytes` to `iri`: as they are when they are UTF-8 text (a model or the
        /// registry), and as `encoding=base64` when they are not (a sealed model).
        pub(super) fn write(&self, iri: &Iri, bytes: &[u8]) -> Result<(), StoreError> {
            let request = Request::new(Verb::Sink, iri.clone());
            let request = if std::str::from_utf8(bytes).is_ok() {
                request.with_arg("content", ArgRef::Inline(bytes.to_vec()))
            } else {
                use base64::Engine as _;
                let text = base64::engine::general_purpose::STANDARD.encode(bytes);
                request
                    .with_arg("content", ArgRef::Inline(text.into_bytes()))
                    .with_arg("encoding", ArgRef::Inline(b"base64".to_vec()))
            };
            self.issue(request)
                .map(drop)
                .map_err(|e| StoreError(format!("write {}: {e}", iri.as_str())))
        }

        pub(super) fn register(&self, model: &str) -> Result<(), StoreError> {
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
            if bytes.len() > MAX_STORED_BYTES {
                return Err(StoreError(format!(
                    "{} is past the bound of {MAX_STORED_BYTES} bytes",
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

        fn registered(&self) -> Result<Vec<String>, StoreError> {
            Ok(self
                .registry
                .lock()
                .expect("registry lock")
                .iter()
                .cloned()
                .collect())
        }
    }
}

#[cfg(not(target_family = "wasm"))]
mod dir {
    use std::fs::{File, OpenOptions, TryLockError};
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex};

    use ikigai_core::{Capability, Kernel};

    use super::{ResourceStore, Store, StoreError};

    /// The lock file's name.
    const LOCK_FILE: &str = "LOCK";
    const PENDING: &str = "PENDING";

    /// A state directory on disk: a [`ResourceStore`] over a private `ikigai-fs` mount
    /// (`urn:file:`) jailed to the directory, and the directory's one-writer lock. See the
    /// module documentation.
    pub struct DirStore {
        dir: PathBuf,
        /// Held, never read: the OS releases the lock when the file closes, so a crashed host
        /// never leaves the directory locked.
        _lock: File,
        models: ResourceStore,
        transaction: Mutex<()>,
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
            let store = DirStore {
                dir,
                _lock: lock,
                models,
                transaction: Mutex::new(()),
            };
            store.recover()?;
            Ok(store)
        }

        // A pending creation is committed once its name is in the registry. Before that,
        // discard it; afterward, finish writing the model before allowing any operation.
        fn recover(&self) -> Result<(), StoreError> {
            let path = self.dir.join(PENDING);
            // ambient: fs: the journal belongs to the locked state directory
            #[allow(clippy::disallowed_methods)]
            let data = match std::fs::read(&path) {
                Ok(data) => data,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
                Err(e) => return Err(StoreError(format!("read {}: {e}", path.display()))),
            };
            let split = data
                .iter()
                .position(|b| *b == b'\n')
                .ok_or_else(|| StoreError(format!("read {}: invalid journal", path.display())))?;
            let model = std::str::from_utf8(&data[..split])
                .map_err(|e| StoreError(format!("read {}: {e}", path.display())))?;
            if self.models.registered()?.iter().any(|name| name == model) {
                self.models
                    .write(&self.models.model_iri(model)?, &data[split + 1..])
                    .map_err(|e| in_dir(&self.dir, e))?;
                self.sync_dir()?;
            }
            // ambient: fs: remove the completed or uncommitted journal under the lock
            #[allow(clippy::disallowed_methods)]
            std::fs::remove_file(&path)
                .map_err(|e| StoreError(format!("remove {}: {e}", path.display())))?;
            self.sync_dir()
        }

        fn sync_dir(&self) -> Result<(), StoreError> {
            #[cfg(unix)]
            {
                // ambient: fs: persist directory entries in the locked state directory
                #[allow(clippy::disallowed_methods)]
                File::open(&self.dir)
                    .and_then(|f| f.sync_all())
                    .map_err(|e| StoreError(format!("sync {}: {e}", self.dir.display())))?;
            }
            Ok(())
        }

        /// The directory.
        pub fn dir(&self) -> &Path {
            &self.dir
        }
    }

    impl Store for DirStore {
        fn load(&self, model: &str) -> Result<Option<Vec<u8>>, StoreError> {
            let _guard = self.transaction.lock().expect("transaction lock");
            self.recover()?;
            self.models.load(model).map_err(|e| in_dir(&self.dir, e))
        }

        fn save(&self, model: &str, bytes: &[u8]) -> Result<(), StoreError> {
            let _guard = self.transaction.lock().expect("transaction lock");
            self.recover()?;
            if self.models.registered()?.iter().any(|name| name == model) {
                return self
                    .models
                    .save(model, bytes)
                    .map_err(|e| in_dir(&self.dir, e));
            }
            // Validate the target before recording it. The journal is staged and synced by
            // the same Sink as model files, then the registry is the commit point.
            let target = self.models.model_iri(model)?;
            if model.contains('\n') {
                return Err(StoreError("model name contains a newline".into()));
            }
            let mut pending = Vec::with_capacity(model.len() + 1 + bytes.len());
            pending.extend_from_slice(model.as_bytes());
            pending.push(b'\n');
            pending.extend_from_slice(bytes);
            self.models
                .write(&self.models.iri(PENDING)?, &pending)
                .map_err(|e| in_dir(&self.dir, e))?;
            self.sync_dir()?;
            self.models
                .register(model)
                .map_err(|e| in_dir(&self.dir, e))?;
            self.sync_dir()?;
            self.models
                .write(&target, bytes)
                .map_err(|e| in_dir(&self.dir, e))?;
            self.sync_dir()?;
            self.recover()
        }

        fn registered(&self) -> Result<Vec<String>, StoreError> {
            let _guard = self.transaction.lock().expect("transaction lock");
            self.recover()?;
            self.models.registered()
        }
    }

    /// A resource error, saying which directory `urn:file:` is jailed to (the error itself
    /// names only the IRI, never a host path).
    fn in_dir(dir: &Path, e: StoreError) -> StoreError {
        StoreError(format!("{e} (urn:file: is {})", dir.display()))
    }

    /// Create the directory `0700` if needed (and make it `0700` if it is not), then take `<dir>/LOCK` exclusively and write this
    /// process's id into it. `ikigai-fs` has no lock, and the one-writer rule keeps two
    /// processes from each training a copy of
    /// a model and losing the other's work.
    fn claim(dir: &Path) -> Result<File, StoreError> {
        private_dir(dir).map_err(|e| StoreError(format!("create {}: {e}", dir.display())))?;
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

    /// Create `dir` (and its missing parents), and on unix make `dir` itself `0700`: the files
    /// the jail writes in it take the process's umask, so the directory is what closes them to
    /// every other account.
    fn private_dir(dir: &Path) -> std::io::Result<()> {
        #[cfg(unix)]
        use std::os::unix::fs::{DirBuilderExt, PermissionsExt};

        let mut builder = std::fs::DirBuilder::new();
        builder.recursive(true);
        #[cfg(unix)]
        builder.mode(0o700);
        // ikigai-fs has no resource to set the jail root's permissions.
        #[allow(clippy::disallowed_methods)] // ambient: fs: create the jail root at 0700 on Unix
        let created = builder.create(dir);
        created?;
        #[cfg(unix)]
        {
            #[allow(clippy::disallowed_methods)] // ambient: fs: check the jail root's mode
            let mode = std::fs::metadata(dir)?.permissions().mode() & 0o7777;
            if mode != 0o700 {
                #[allow(clippy::disallowed_methods)] // ambient: fs: enforce owner-only mode 0700
                let secured = std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700));
                secured?;
            }
        }
        Ok(())
    }
}
