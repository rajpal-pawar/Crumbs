use ort;
fn main() {
    let x = ort::init().with_name("test").commit();
    let _: () = x; // This will trigger a type mismatch error showing the actual type!
}
