use std::{
    fs,
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
};
static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

struct Model(std::path::PathBuf);
impl Model {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "connect4-eval-{}-{}.bin",
            std::process::id(),
            NEXT_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let mut bytes = b"C4V2".to_vec();
        for n in [3u32, 84, 1, 1] {
            bytes.extend(n.to_le_bytes());
        }
        bytes.extend(vec![0u8; 87 * 4]);
        fs::write(&path, bytes).unwrap();
        Self(path)
    }
}
impl Drop for Model {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

#[test]
fn progress_is_separate_from_results_and_can_be_disabled() {
    let model = Model::new();
    let run = |quiet: bool| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_connect-4"));
        command
            .arg("eval")
            .arg(&model.0)
            .args(["--games", "12", "--threads", "3"]);
        if quiet {
            command.arg("--no-progress");
        }
        command.output().unwrap()
    };
    let normal = run(false);
    assert!(normal.status.success());
    let stdout = String::from_utf8(normal.stdout).unwrap();
    let stderr = String::from_utf8(normal.stderr).unwrap();
    assert!(
        stdout.starts_with('{') && stdout.contains("\"first\"") && stdout.contains("\"total\"")
    );
    assert!(!stdout.contains("ETA"));
    assert!(stderr.contains("0/12") && stderr.contains("12/12 (100%)") && stderr.contains("ETA"));
    let quiet = run(true);
    assert!(quiet.status.success());
    assert!(quiet.stderr.is_empty());
    // Exclude timing fields when comparing identical seeded evaluations.
    assert_eq!(
        stdout.split("\"seconds\"").next(),
        String::from_utf8(quiet.stdout)
            .unwrap()
            .split("\"seconds\"")
            .next()
    );
}

#[test]
fn malformed_arguments_fail_without_starting_games() {
    for args in [
        vec!["eval"],
        vec!["eval", "missing.bin", "--games", "3"],
        vec!["eval", "missing.bin", "--threads", "0"],
        vec!["eval", "missing.bin", "--opponent", "checkpoint"],
    ] {
        let result = Command::new(env!("CARGO_BIN_EXE_connect-4"))
            .args(args)
            .output()
            .unwrap();
        assert_eq!(result.status.code(), Some(2));
        assert!(result.stdout.is_empty());
    }
}
