#[cfg(not(target_family = "wasm"))]
#[test]
fn failed_first_save_does_not_leave_model_bytes() {
    use ikigai_bayes::{
        model::Model,
        store::{DirStore, Store},
    };
    let dir = std::env::current_dir()
        .unwrap()
        .join("target")
        .join(format!("audit-store-{}", std::process::id()));
    let store = DirStore::open(&dir).unwrap();
    std::fs::create_dir(dir.join("models")).unwrap();
    let result = store.save("contact", &Model::new("contact").to_bytes());
    let exists = dir.join("contact.json").exists();
    drop(store);
    std::fs::remove_dir_all(&dir).unwrap();
    assert!(result.is_err());
    assert!(!exists);
}
