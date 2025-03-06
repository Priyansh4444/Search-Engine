use porter_stemmer::stem;
use regex::Regex;
pub struct Tokenizer {}

impl Tokenizer {
    pub fn new() -> Tokenizer {
        Tokenizer {}
    }
    pub fn tokenize(&self, token: &str) -> Vec<String> {
        let pattern = Regex::new(r"[^a-zA-Z0-9]+").unwrap();
        pattern
            .split(token)
            .map(|s| s.to_lowercase())
            .map(|s| self.porter_stemmer(&s))
            .filter(|s: &String| !s.is_empty())
            .collect::<Vec<String>>()
    }

    pub fn porter_stemmer(&self, token: &str) -> String {
        stem(token)
    }
}
