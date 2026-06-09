//! Phase 5: linking emitted objects into executables.
//!
//! Xia links directly against libc with no wrapper runtime. On Windows the
//! object is linked with `lld-link` (bundled with the LLVM dev package, or on
//! PATH) against the MSVC + Windows SDK import libraries; on Unix hosts we
//! drive `cc`, which knows the platform's CRT startup files.

use std::path::{Path, PathBuf};
use std::process::Command;

pub fn link(obj: &Path, exe: &Path) -> Result<(), String> {
    if cfg!(windows) {
        link_windows(obj, exe)
    } else {
        link_unix(obj, exe)
    }
}

fn link_unix(obj: &Path, exe: &Path) -> Result<(), String> {
    let out = Command::new("cc")
        .arg(obj)
        .arg("-o")
        .arg(exe)
        .output()
        .map_err(|e| format!("failed to run cc: {e}"))?;
    if out.status.success() {
        Ok(())
    } else {
        Err(format!(
            "cc failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        ))
    }
}

fn link_windows(obj: &Path, exe: &Path) -> Result<(), String> {
    let lld = find_lld_link().ok_or(
        "lld-link not found: install LLVM or set LLVM_SYS_181_PREFIX to an LLVM tree",
    )?;
    let libpaths = windows_lib_paths()?;

    let mut cmd = Command::new(lld);
    cmd.arg("/nologo")
        .arg("/subsystem:console")
        .arg(obj)
        .arg(format!("/out:{}", exe.display()));
    for p in libpaths {
        cmd.arg(format!("/libpath:{}", p.display()));
    }
    // Dynamic CRT (/MD) set; legacy_stdio_definitions provides printf, which
    // the UCRT otherwise defines as an inline function.
    for lib in [
        "msvcrt.lib",
        "vcruntime.lib",
        "ucrt.lib",
        "legacy_stdio_definitions.lib",
        "kernel32.lib",
    ] {
        cmd.arg(lib);
    }

    let out = cmd
        .output()
        .map_err(|e| format!("failed to run lld-link: {e}"))?;
    if out.status.success() {
        Ok(())
    } else {
        Err(format!(
            "lld-link failed:\n{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ))
    }
}

fn find_lld_link() -> Option<PathBuf> {
    if let Ok(prefix) = std::env::var("LLVM_SYS_181_PREFIX") {
        let candidate = Path::new(&prefix).join("bin").join("lld-link.exe");
        if candidate.exists() {
            return Some(candidate);
        }
    }
    // Fall back to PATH.
    which("lld-link.exe").or_else(|| which("lld-link"))
}

fn which(name: &str) -> Option<PathBuf> {
    std::env::var_os("PATH")?
        .to_string_lossy()
        .split(';')
        .map(|dir| Path::new(dir).join(name))
        .find(|p| p.exists())
}

/// Locate the MSVC toolchain and Windows SDK import libraries (x64).
fn windows_lib_paths() -> Result<Vec<PathBuf>, String> {
    let mut paths = Vec::new();

    // MSVC: <VS>\VC\Tools\MSVC\<ver>\lib\x64, newest VS / newest toolset.
    let vswhere = Path::new(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe");
    if vswhere.exists() {
        let out = Command::new(vswhere)
            .args([
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ])
            .output()
            .map_err(|e| format!("vswhere failed: {e}"))?;
        let vs_root = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !vs_root.is_empty() {
            let msvc_root = Path::new(&vs_root).join(r"VC\Tools\MSVC");
            if let Some(toolset) = newest_subdir(&msvc_root) {
                paths.push(toolset.join(r"lib\x64"));
            }
        }
    }

    // Windows SDK: <kits>\10\Lib\<ver>\{ucrt,um}\x64
    let sdk_lib = Path::new(r"C:\Program Files (x86)\Windows Kits\10\Lib");
    if let Some(ver) = newest_subdir(sdk_lib) {
        paths.push(ver.join(r"ucrt\x64"));
        paths.push(ver.join(r"um\x64"));
    }

    let missing: Vec<_> = paths.iter().filter(|p| !p.exists()).collect();
    if paths.len() < 3 || !missing.is_empty() {
        return Err(
            "could not locate MSVC / Windows SDK libraries; install Visual Studio Build Tools"
                .into(),
        );
    }
    Ok(paths)
}

fn newest_subdir(dir: &Path) -> Option<PathBuf> {
    let mut subdirs: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    subdirs.sort();
    subdirs.pop()
}
