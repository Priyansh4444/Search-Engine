use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::thread::{self};
use std::time;
use std::{
    fs::File,
    io::{self, Write},
};

use crate::id_book::IDBookElement;
use crate::{file_skip_list, tokenizer::Tokenizer};

pub const TOTAL_DOCUMENT_COUNT: u16 = 46843;
// Precomputed average document length - should be calculated once at initialization
pub const AVG_DOC_LENGTH: f64 = 1000.0; // Replace with actual average calculated from corpus

// Thread pool size - adjust based on your system's CPU cores
const THREAD_POOL_SIZE: usize = 4;

pub struct SearchEngine {
    query: String,
    tokens: Vec<String>,
    skiplists: Arc<Vec<Vec<file_skip_list::FileSkip>>>,
    // Cache the average document length
    avg_doc_length: f64,
}

impl SearchEngine {
    pub fn new() -> Self {
        // Load skiplists in a more compact way
        let mut skiplists = Vec::with_capacity(36);

        // Load digit skiplists (0-9)
        for c in b'0'..=b'9' {
            skiplists.push(file_skip_list::FileSkip::read_skip_list(c as char));
        }

        // Load alphabetic skiplists (a-z)
        for c in b'a'..=b'z' {
            skiplists.push(file_skip_list::FileSkip::read_skip_list(c as char));
        }

        // Calculate average document length (should ideally be done once and stored)
        // For now using a constant, but in practice this should be computed

        Self {
            query: String::new(),
            tokens: Vec::new(),
            skiplists: Arc::new(skiplists),
            avg_doc_length: AVG_DOC_LENGTH,
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

        // Early exit for empty queries
        if self.tokens.is_empty() {
            return (Vec::new(), 0);
        }

        // Process tokens and gather candidates
        let candidates = self.process_tokens();

        if candidates.is_empty() {
            return (Vec::new(), 0);
        }

        // Optimize: Filter documents more efficiently
        let filtered_candidates = self.filter_candidates(candidates);

        // Get documents that match all query terms
        let doc_ids = self.get_matching_doc_ids(&filtered_candidates);

        if doc_ids.is_empty() {
            return (Vec::new(), 0);
        }

        // Score documents
        let scored_docs = self.score_documents(doc_ids, &filtered_candidates);

        // Sort and extract results
        let final_time = time.elapsed().as_millis();
        let results = self.format_results(scored_docs, final_time);

        (results, final_time)
    }

    // Process tokens in parallel and return candidates
    fn process_tokens(&self) -> Vec<Candidate> {
        let candidates = Arc::new(Mutex::new(Vec::with_capacity(self.tokens.len())));
        let mut handles = Vec::with_capacity(self.tokens.len());

        for token in &self.tokens {
            let candidates = Arc::clone(&candidates);
            let skiplists = Arc::clone(&self.skiplists);
            let token = token.clone();

            let handle = thread::spawn(move || {
                // Get first character to determine which skiplist to use
                let first_char = token.chars().next().unwrap();
                let first_char_index = if first_char.is_ascii_digit() {
                    (first_char as u8 - b'0') as usize
                } else {
                    (first_char as u8 - b'a') as usize + 10
                };

                // Find offsets in the skiplist
                let offset_range =
                    file_skip_list::FileSkip::find_skip_entry(&skiplists[first_char_index], &token);

                // Generate file path and create candidate
                let file_path = format!("inverted_index/merged/{}.txt", first_char);
                let mut candidate = Candidate::new(token.to_string());

                // Load postings from file
                if let Ok(file) = File::open(&file_path) {
                    let postings =
                        file_skip_list::get_postings_from_offset_range(&file, offset_range, &token);

                    // Update candidate with posting information
                    let posting_length = postings.postings.len() as u16;
                    for single_posting in postings.postings {
                        candidate
                            .update_freq(single_posting.doc_id, single_posting.term_freq as u32);
                    }
                    candidate.update_posting_length(posting_length);
                } else {
                    println!("Warning: Could not open index file for '{}'", first_char);
                }

                // Add candidate to shared list
                let mut candidates = candidates.lock().unwrap();
                candidates.push(candidate);
            });

            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // Extract candidates from Arc<Mutex>
        Arc::try_unwrap(candidates).unwrap().into_inner().unwrap()
    }

    // Filter candidates to find documents matching all terms
    fn filter_candidates(&self, mut candidates: Vec<Candidate>) -> Vec<Candidate> {
        // Early exit if no candidates
        if candidates.is_empty() {
            return candidates;
        }

        // Sort by selectivity (smaller posting lists first for faster intersection)
        candidates.sort_by_key(|c| c.posting_length);

        // Get intersection of document sets
        let mut common_docs: Option<HashSet<u16>> = None;

        for candidate in &candidates {
            let current_docs: HashSet<u16> = candidate.doc_ids.keys().cloned().collect();

            match &mut common_docs {
                None => common_docs = Some(current_docs),
                Some(docs) => {
                    // Use smaller set for iteration in intersection
                    if docs.len() > current_docs.len() {
                        let temp_docs = docs.clone();
                        docs.clear();
                        for &doc_id in &current_docs {
                            if temp_docs.contains(&doc_id) {
                                docs.insert(doc_id);
                            }
                        }
                    } else {
                        docs.retain(|doc_id| current_docs.contains(doc_id));
                    }

                    // Early termination for empty intersection
                    if docs.is_empty() {
                        break;
                    }
                }
            }
        }

        // Filter candidates to only keep common documents
        let common_docs = common_docs.unwrap_or_default();
        if !common_docs.is_empty() {
            for candidate in &mut candidates {
                candidate.doc_ids.retain(|k, _| common_docs.contains(k));
            }
        }

        candidates
    }

    // Get the set of document IDs that match the query
    fn get_matching_doc_ids(&self, candidates: &[Candidate]) -> Vec<u16> {
        let mut doc_ids = HashSet::new();

        // Union all document IDs
        for candidate in candidates {
            for &doc_id in candidate.doc_ids.keys() {
                doc_ids.insert(doc_id);
            }
        }

        doc_ids.into_iter().collect()
    }

    // Score documents using the improved TF-IDF algorithm
    fn score_documents(&self, doc_ids: Vec<u16>, candidates: &[Candidate]) -> Vec<(u16, f64)> {
        // For small document sets, process directly
        if doc_ids.len() < 100 {
            return self.score_documents_direct(doc_ids, candidates);
        }

        // For larger sets, use batched parallel processing
        self.score_documents_parallel(doc_ids, candidates)
    }

    // Direct scoring for small document sets
    fn score_documents_direct(
        &self,
        doc_ids: Vec<u16>,
        candidates: &[Candidate],
    ) -> Vec<(u16, f64)> {
        let mut scores = Vec::with_capacity(doc_ids.len());

        for doc_id in doc_ids {
            let mut score = 0.0;
            let doc = IDBookElement::get_doc_from_id(doc_id);

            for candidate in candidates {
                if let Some(&term_freq) = candidate.doc_ids.get(&doc_id) {
                    score += improved_tf_idf(
                        term_freq as u16,
                        candidate.posting_length,
                        doc.token_count,
                        candidates.len(),
                        self.avg_doc_length,
                    );
                }
            }

            scores.push((doc_id, score));
        }

        // Sort by score in descending order
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
    }

    // Parallel scoring for larger document sets
    fn score_documents_parallel(
        &self,
        doc_ids: Vec<u16>,
        candidates: &[Candidate],
    ) -> Vec<(u16, f64)> {
        let doc_count = doc_ids.len();
        let batch_size = (doc_count + THREAD_POOL_SIZE - 1) / THREAD_POOL_SIZE;

        let mut batched_ids = Vec::with_capacity(THREAD_POOL_SIZE);
        for i in 0..THREAD_POOL_SIZE {
            let start = i * batch_size;
            let end = std::cmp::min(start + batch_size, doc_count);
            if start < end {
                batched_ids.push(doc_ids[start..end].to_vec());
            }
        }

        let candidates_arc = Arc::new(candidates.to_vec());
        let score_results = Arc::new(Mutex::new(Vec::with_capacity(doc_count)));
        let mut batch_handles = Vec::with_capacity(batched_ids.len());
        let avg_doc_length = self.avg_doc_length;
        let query_term_count = candidates.len();

        // Process each batch in parallel
        for batch in batched_ids {
            let candidates_arc = Arc::clone(&candidates_arc);
            let score_results = Arc::clone(&score_results);

            let handle = thread::spawn(move || {
                let mut batch_scores = Vec::with_capacity(batch.len());

                for doc_id in batch {
                    let mut doc_score = 0.0;
                    let doc = IDBookElement::get_doc_from_id(doc_id);

                    for candidate in candidates_arc.iter() {
                        if let Some(&term_freq) = candidate.doc_ids.get(&doc_id) {
                            doc_score += improved_tf_idf(
                                term_freq as u16,
                                candidate.posting_length,
                                doc.token_count,
                                query_term_count,
                                avg_doc_length,
                            );
                        }
                    }

                    batch_scores.push((doc_id, doc_score));
                }

                let mut results = score_results.lock().unwrap();
                results.extend(batch_scores);
            });

            batch_handles.push(handle);
        }

        // Wait for all batches to complete
        for handle in batch_handles {
            handle.join().unwrap();
        }

        // Get final results
        let mut scored_docs = Arc::try_unwrap(score_results)
            .unwrap()
            .into_inner()
            .unwrap();

        // Sort by score in descending order
        scored_docs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored_docs
    }

    // Format the results for output
    fn format_results(&self, scored_docs: Vec<(u16, f64)>, elapsed_time: u128) -> Vec<String> {
        println!("Search took: {}ms", elapsed_time);

        let mut results = Vec::with_capacity(10);
        for (doc_id, score) in scored_docs.iter().take(10) {
            let doc = IDBookElement::get_doc_from_id(*doc_id);
            println!(
                "{}|> {}: {} (Score: {:.4})",
                doc_id,
                doc.url,
                doc.path.display(),
                score
            );
            results.push(doc.url.clone());
        }

        results
    }
}

#[derive(Debug, Clone)]
pub struct Candidate {
    pub term: String,
    pub posting_length: u16,
    pub doc_ids: HashMap<u16, u32>, // doc_id -> term_freq
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
}

// Improved TF-IDF scoring function with BM25-inspired normalization
pub fn improved_tf_idf(
    term_freq: u16,
    posting_length: u16,
    doc_length: u32,
    query_term_count: usize,
    avg_doc_length: f64,
) -> f64 {
    // BM25 parameters (can be tuned)
    let k1 = 1.2; // Controls term frequency saturation
    let b = 0.75; // Controls document length normalization

    // Document length normalization factor
    let doc_length_ratio = doc_length as f64 / avg_doc_length;

    // TF component with saturation and length normalization
    let tf_component = (term_freq as f64 * (k1 + 1.0))
        / (term_freq as f64 + k1 * (1.0 - b + b * doc_length_ratio));

    // IDF component (with smoothing to prevent negative values)
    let idf = f64::max(
        0.0,
        f64::ln(
            (TOTAL_DOCUMENT_COUNT as f64 - posting_length as f64 + 0.5)
                / (posting_length as f64 + 0.5),
        ),
    );

    // Normalize by query length to ensure fair comparison between queries of different lengths
    let query_norm_factor = 1.0 / (query_term_count as f64).sqrt();

    // Calculate final score
    tf_component * idf * query_norm_factor
}
