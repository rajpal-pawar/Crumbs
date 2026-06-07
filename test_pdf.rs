use pdf_extract;

fn main() {
    let path = "/home/rajpalsinghpanwar/Crumbs/data/test_files/IV Sem syllabus IT.pdf";
    match pdf_extract::extract_text(path) {
        Ok(text) => println!("Success: {} chars", text.len()),
        Err(e) => println!("Error: {}", e),
    }
}
