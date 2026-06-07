// build.rs — Required by tauri-build to emit the correct linker args and
// generate the resource manifests (icons, capabilities, etc.) that Tauri
// expects to find at compile time.
fn main() {
    tauri_build::build()
}
