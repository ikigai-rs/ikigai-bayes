use ikigai_bayes::tokenize::tokenize;

#[test]
fn scheme_in_path_keeps_bare_www_authority() {
    let tokens = tokenize("www.example.com/https://other.example", &[]).unwrap();
    assert!(tokens.contains("url-scheme:none"), "{tokens:?}");
    assert!(tokens.contains("url-host:www.example.com"), "{tokens:?}");
}
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
