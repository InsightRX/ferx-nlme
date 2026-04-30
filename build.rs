fn main() {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    println!("cargo:rustc-env=FERX_BUILD_TIMESTAMP={}", timestamp);

    let has_autodiff = std::env::var("CARGO_FEATURE_AUTODIFF").is_ok();
    let has_ci = std::env::var("CARGO_FEATURE_CI").is_ok();
    let variant = match (has_autodiff, has_ci) {
        (true, _) => "autodiff",
        (false, true) => "ci",
        (false, false) => "unknown",
    };
    println!("cargo:rustc-env=FERX_BUILD_VARIANT={}", variant);

    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let output = std::process::Command::new(&rustc)
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=FERX_RUSTC_VERSION={}", output.trim());

    let profile = std::env::var("PROFILE").unwrap_or_else(|_| "unknown".into());
    println!("cargo:rustc-env=FERX_BUILD_PROFILE={}", profile);

    println!("cargo:rerun-if-changed=Cargo.toml");
    // Re-run when any input we read above changes, so the embedded metadata
    // does not go stale across feature/profile/toolchain switches.
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_AUTODIFF");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CI");
    println!("cargo:rerun-if-env-changed=PROFILE");
    println!("cargo:rerun-if-env-changed=RUSTC");
}
