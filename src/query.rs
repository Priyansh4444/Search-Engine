use std::time;
use std::{
    collections::HashMap,
    io::{self, Write},
};

use crate::id_book::IDBookElement;
use crate::{file_skip_list, tokenizer::Tokenizer};
use std::fs::File;
use std::sync::{Arc, Mutex};
use std::thread;

pub struct SearchEngine {
    query: String,
    tokens: Vec<String>,
}

impl SearchEngine {
    pub fn new() -> Self {
        Self {
            query: String::new(),
            tokens: Vec::new(),
        }
    }

    pub fn get_query(&mut self) {
        self.query.clear();
        print!("Enter your search query: ");
        io::stdout().flush().unwrap();
        io::stdin()
            .read_line(&mut self.query)
            .expect("Failed to read query");
        self.query = self.query.trim().to_string();
        self.tokens = Tokenizer::new().tokenize(&self.query);
    }

    pub fn set_query(&mut self, query: String) {
        self.query = query;
        self.tokens = Tokenizer::new().tokenize(&self.query);
    }

    pub fn search(&self) -> (Vec<String>, u128) {
        let time = time::Instant::now();
        println!("Searching for: \"{}\"", self.query);
        println!("Tokens: {:?}", self.tokens);
        // this will be shared across threads for adding candidates
        let candidates = Arc::new(Mutex::new(Vec::with_capacity(self.tokens.len())));
        // create a thread for each token to search for the token in the inverted index
        let mut handles = vec![];

        for token in self.tokens.iter() {
            let candidates = Arc::clone(&candidates);
            // spawn a thread for each token
            // cloning since the "token" would otherwise be consumed by the thread and that reference lifetime could only work if it were static!
            let token = token.clone();
            let handle = thread::spawn(move || {
                // get the first letter of the token to determine which file to read
                let first_char = token.chars().next().clone().unwrap();
                // go to the file_skip list of the first character
                let skiplist = file_skip_list::FileSkip::read_skip_list(first_char);
                // get which BYTE range it is in between
                let offset_range = file_skip_list::FileSkip::find_skip_entry(&skiplist, &token);
                let file_path = format!("inverted_index/merged/{}.txt", first_char);

                let mut candidate = Candidate::new(token.to_string());
                if let Ok(file) = File::open(&file_path) {
                    // get the postings from the file and update the scorings of the candidates
                    let postings =
                        file_skip_list::get_postings_from_offset_range(&file, offset_range, &token);
                    // Update candidates with the postings data
                    for single_posting in postings.postings {
                        candidate.update_score(single_posting.doc_id, single_posting.term_freq);
                    }
                } else {
                    println!("Warning: Could not open index file for '{}'", first_char);
                }
                let mut candidates = candidates.lock().unwrap();
                candidates.push(candidate);
            });
            handles.push(handle);
        }
        // after each thread is done, join them
        for handle in handles {
            handle.join().unwrap();
        }
        // unwrap the candidates from the mutex lock
        let mut candidates = Arc::try_unwrap(candidates).unwrap().into_inner().unwrap();
        // sort the candidates by the number of documents they appear in with smallest first
        candidates.sort_by(|a, b| a.doc_ids.len().cmp(&b.doc_ids.len()));
        // then filter candidates that only have all the tokens
        let mut boolean_and_candidates: HashMap<&u16, &u16> =
            candidates[0].doc_ids.iter().collect();
        for candidate in candidates.iter().skip(1) {
            boolean_and_candidates.retain(|doc_id, _| candidate.doc_ids.contains_key(doc_id));
        }

        // Sort candidates by score (term frequency) in descending order
        let mut sorted_candidates: Vec<(&u16, &u16)> = boolean_and_candidates.into_iter().collect();
        sorted_candidates.sort_by(|a, b| b.1.cmp(a.1));
        let mut results = Vec::new();
        for (doc_id, term_freq) in sorted_candidates.iter().take(5) {
            let doc = IDBookElement::get_doc_from_id(**doc_id);
            println!(
                "{}|> {}: {} (Score: {})",
                doc_id,
                doc.url,
                doc.path.display(),
                term_freq
            );
            results.push(doc.url.clone());
        }
        println!("Search took: {}ms", time.elapsed().as_millis());
        (results, time.elapsed().as_millis())
    }
}

#[derive(Debug)]
pub struct Candidate {
    pub term: String,
    pub doc_ids: HashMap<u16, u16>, // for each doc_id, the term frequency
}

impl Candidate {
    pub fn new(token: String) -> Self {
        Self {
            term: token,
            doc_ids: HashMap::new(),
        }
    }
    pub fn update_score(&mut self, doc_id: u16, term_freq: u16) {
        self.doc_ids.insert(doc_id, term_freq);
    }
}
