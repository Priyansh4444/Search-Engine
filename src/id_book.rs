use crate::index_builder::IDBOOK_PATH;
use std::{
    io::{BufRead, Seek},
    path::PathBuf,
};
pub struct IDBookElement {
    pub id: u16,
    pub url: String,
    pub path: PathBuf,
}

impl IDBookElement {
    pub fn new(id: u16, url: String, path: PathBuf) -> Self {
        Self { id, url, path }
    }

    pub fn get_domain(&self) -> String {
        let parts: Vec<&str> = self.url.splitn(4, '/').collect();
        if parts.len() >= 3 {
            format!("{}//{}", parts[0], parts[2])
        } else {
            self.url.to_string()
        }
    }

    pub fn idbook_element_from_string(id: u16, line: &str) -> Self {
        let mut parts = line.splitn(2, '|');
        let url = parts.next().unwrap().trim().to_string();
        let path = PathBuf::from(parts.next().unwrap().trim().to_string());
        Self::new(id, url, path)
    }

    pub fn get_doc_from_id(id: u16) -> Self {
        let buffer = std::fs::File::open(IDBOOK_PATH).unwrap();
        // skip 400  * (id - 1) bytes
        let mut reader = std::io::BufReader::new(buffer);
        reader
            .seek(std::io::SeekFrom::Start(400 * (id as u64 - 1)))
            .unwrap();
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        Self::idbook_element_from_string(id, &line)
    }
}
