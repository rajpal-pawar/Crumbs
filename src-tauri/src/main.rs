// main.rs — Tauri application entry point.
//
// On Windows this binary is compiled as a "windows" subsystem application so
// that no console window flashes when the user launches Crumbs.
// The attribute is ignored on other platforms.

#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

fn main() {
    crumbs_tauri_lib::run();
}
