use tauri_plugin_global_shortcut::{Code, Modifiers, Shortcut, ShortcutState};

fn main() {
    let ctrl_shift_space = Shortcut::new(Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::Space);
    let _ = tauri_plugin_global_shortcut::Builder::new()
        .with_shortcut(ctrl_shift_space)
        .unwrap()
        .with_handler(|_app, _shortcut, _event| {})
        .build();
}
