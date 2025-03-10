use std::collections::HashSet;
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

pub const TOTAL_DOCUMENT_COUNT: u16 = 46843;

pub struct SearchEngine {
    query: String,
    tokens: Vec<String>,
    skiplists: Arc<Vec<Vec<file_skip_list::FileSkip>>>,
}

impl SearchEngine {
    pub fn new() -> Self {
        let mut skiplists = Vec::new();
        for i in 0..=9 {
            let skiplist = file_skip_list::FileSkip::read_skip_list((b'0' + i) as char);
            skiplists.push(skiplist);
        }
        for i in 0..26 {
            let skiplist = file_skip_list::FileSkip::read_skip_list((b'a' + i) as char);
            skiplists.push(skiplist);
        }
        Self {
            query: String::new(),
            tokens: Vec::new(),
            skiplists: Arc::new(skiplists),
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

        // This will be shared across threads for adding candidates
        let candidates = Arc::new(Mutex::new(Vec::with_capacity(self.tokens.len())));
        let mut handles = vec![];

        for token in self.tokens.iter() {
            let candidates = Arc::clone(&candidates);
            let skiplists = Arc::clone(&self.skiplists);
            let token = token.clone();

            let handle = thread::spawn(move || {
                let first_char = token.chars().next().unwrap();
                let first_char_index = if first_char.is_ascii_digit() {
                    (first_char as u8 - b'0') as usize
                } else {
                    (first_char as u8 - b'a') as usize + 10
                };

                let offset_range =
                    file_skip_list::FileSkip::find_skip_entry(&skiplists[first_char_index], &token);

                let file_path = format!("inverted_index/merged/{}.txt", first_char);

                let mut candidate = Candidate::new(token.to_string());
                if let Ok(file) = File::open(&file_path) {
                    let postings =
                        file_skip_list::get_postings_from_offset_range(&file, offset_range, &token);
                    let posting_length = postings.postings.len() as u16;
                    for single_posting in postings.postings {
                        candidate
                            .update_freq(single_posting.doc_id, single_posting.term_freq as u32);
                    }
                    candidate.update_posting_length(posting_length);
                } else {
                    println!("Warning: Could not open index file for '{}'", first_char);
                }
                let mut candidates = candidates.lock().unwrap();
                candidates.push(candidate);
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        if self.tokens.len() == 0 {
            return (Vec::new(), 0);
        }

        let mut candidates = Arc::try_unwrap(candidates).unwrap().into_inner().unwrap();
        if candidates.len() == 0 {
            return (Vec::new(), 0);
        }

        filter_and_sort_candiadtes(&mut candidates);

        let mut results = Vec::new();

        let mut combined_scores: HashSet<u16> = HashSet::new();
        for candidate in &candidates {
            for doc_id in candidate.doc_ids.keys() {
                combined_scores.insert(*doc_id);
            }
        }

        let doc_ids: Vec<u16> = combined_scores.into_iter().collect();
        let doc_count = doc_ids.len();
        let batch_size = (doc_count + 4) / 5;

        let mut batched_ids = Vec::new();
        for i in 0..5 {
            let start = i * batch_size;
            let end = std::cmp::min(start + batch_size, doc_count);
            if start < end {
                batched_ids.push(doc_ids[start..end].to_vec());
            }
        }

        let candidates_arc = Arc::new(candidates);
        let score_results = Arc::new(Mutex::new(HashMap::new()));
        let mut batch_handles = vec![];

        // Process each batch in parallel
        for batch in batched_ids {
            let candidates_arc = Arc::clone(&candidates_arc);
            let score_results = Arc::clone(&score_results);

            let handle = thread::spawn(move || {
                let mut batch_scores = HashMap::new();

                for doc_id in batch {
                    let mut doc_score = 0.0;
                    for candidate in candidates_arc.iter() {
                        if let Some(term_freq) = candidate.doc_ids.get(&doc_id) {
                            let total_terms = IDBookElement::get_doc_from_id(doc_id).token_count;
                            let score = scoring_tf_idf(
                                *term_freq as u16,
                                candidate.posting_length,
                                total_terms,
                            );
                            doc_score += score;
                        }
                    }
                    batch_scores.insert(doc_id, doc_score);
                }

                let mut results = score_results.lock().unwrap();
                for (doc_id, score) in batch_scores {
                    results.insert(doc_id, score);
                }
            });

            batch_handles.push(handle);
        }

        // Wait for all batches to complete
        for handle in batch_handles {
            handle.join().unwrap();
        }

        // Convert scores to vector and sort
        let score_map = Arc::try_unwrap(score_results)
            .unwrap()
            .into_inner()
            .unwrap();
        let mut sorted_candidates: Vec<(u16, f64)> = score_map
            .into_iter()
            .map(|(doc_id, score)| (doc_id, score))
            .collect();

        // Sort by score in descending order
        sorted_candidates
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let final_time = time.elapsed().as_millis();
        println!("Search took: {}ms", final_time);
        for (doc_id, score) in sorted_candidates.iter().take(10) {
            let doc = IDBookElement::get_doc_from_id(*doc_id);
            println!(
                "{}|> {}: {} (Score: {})",
                doc_id,
                doc.url,
                doc.path.display(),
                score
            );
            results.push(doc.url.clone());
        }
        (results, final_time)
    }
}

#[derive(Debug)]
pub struct Candidate {
    pub term: String,
    pub posting_length: u16,
    pub doc_ids: HashMap<u16, u32>, // for each doc_id, the tfidf score
}

impl Candidate {
    pub fn new(token: String) -> Self {
        Self {
            term: token,
            posting_length: 0,
            doc_ids: HashMap::new(),
        }
    }

    pub fn update_posting_length(&mut self, posting_length: u16) {
        self.posting_length = posting_length;
    }

    pub fn update_freq(&mut self, doc_id: u16, term_freq: u32) {
        self.doc_ids.insert(doc_id, term_freq);
    }

    pub fn get_score(&self) -> f64 {
        let mut score = 0.0;
        for (doc_id, term_freq) in self.doc_ids.iter() {
            let total_terms_in_document = IDBookElement::get_doc_from_id(*doc_id).token_count;
            score += scoring_tf_idf(
                *term_freq as u16,
                self.posting_length,
                total_terms_in_document,
            );
        }
        score
    }
}

pub fn scoring_tf_idf(term_freq: u16, posting_length: u16, total_terms_in_document: u32) -> f64 {
    // TF: term frequency normalized by document length
    // Higher term_freq should result in higher TF
    let tf: f64 = term_freq as f64 / total_terms_in_document as f64;
    // Logarithmic scaling to dampen the effect of high frequencies
    let tf_scaled: f64 = if tf > 0.0 { 1.0 + f64::log10(tf) } else { 0.0 };
    // IDF: inverse document frequency
    // Lower posting_length (fewer docs with this term) gives higher IDF
    let idf: f64 = f64::log10(TOTAL_DOCUMENT_COUNT as f64 / (posting_length as f64 + 1.0));

    tf_scaled * idf
}

fn filter_and_sort_candiadtes(candidates: &mut Vec<Candidate>) {
    candidates.sort_by(|a: &Candidate, b: &Candidate| a.doc_ids.len().cmp(&b.doc_ids.len()));

    let final_doc_ids = candidates[0]
        .doc_ids
        .keys()
        .cloned()
        .collect::<HashSet<u16>>();
    for candidate in candidates.iter_mut().skip(1) {
        candidate.doc_ids.retain(|k, _| final_doc_ids.contains(k));
    }

    candidates.sort_by(|a: &Candidate, b: &Candidate| {
        let a_score = a.doc_ids.values().sum::<u32>();
        let b_score = b.doc_ids.values().sum::<u32>();
        b_score.cmp(&a_score)
    });
}
