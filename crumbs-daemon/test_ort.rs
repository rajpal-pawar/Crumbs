fn main() {
    println!("Starting");
    let res = std::panic::catch_unwind(|| {
        ort::init().with_name("crumbs-embed").commit()
    });
    println!("Result: {:?}", res);
}
