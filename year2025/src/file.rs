use std::error::Error;
use std::path::PathBuf;

pub fn read_file(filename: &str) -> Result<String, Box<dyn Error>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("inputs")
        .join(filename);
    Ok(std::fs::read_to_string(path)?)
}
