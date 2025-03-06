use crate::postings::Postings;
use crate::tokenizer::Tokenizer;
use std::{
    collections::{BTreeSet, HashMap},
    fs::File,
    io::Write,
};

#[derive(Clone)]
pub struct InvertedIndex {
    index: HashMap<String, Postings>,
    ordered_keys: BTreeSet<String>,
}

#[allow(dead_code)]
impl InvertedIndex {
    pub fn new() -> InvertedIndex {
        InvertedIndex {
            index: HashMap::new(),
            ordered_keys: BTreeSet::new(),
        }
    }

    pub fn insert(&mut self, term: String, doc_id: u16) {
        let postings = self
            .index
            .entry(term.clone())
            .or_insert(Postings::new(term.clone()));
        postings.update_frequency(doc_id);
        self.ordered_keys.insert(term);
    }

    pub fn get_postings(&self, term: &str) -> Option<&Postings> {
        self.index.get(term)
    }

    pub fn get_ordered_keys(&self) -> Vec<String> {
        self.ordered_keys.iter().cloned().collect()
    }
    pub fn merge(&mut self, other: InvertedIndex) {
        for (term, postings) in other.index {
            let self_postings = self
                .index
                .entry(term.clone())
                .or_insert(Postings::new(term));
            for posting in postings.get_postings() {
                self_postings.update_frequency(posting.doc_id);
            }
        }
    }
}

//  *This might also help us with multithreading if we wish to make it multithreaded
// https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf
// page 76, a-f, g-p, q-z get their own inverted indexes and post in their own files
// lets also make one for 0-9
// lets also make one for special characters
pub struct InvertedIndexSplit {
    pub a_f: InvertedIndex,
    pub g_p: InvertedIndex,
    pub q_z: InvertedIndex,
    pub zero_nine: InvertedIndex,
    pub tokenizer: Tokenizer,
}

impl InvertedIndexSplit {
    pub fn new() -> InvertedIndexSplit {
        InvertedIndexSplit {
            a_f: InvertedIndex::new(),
            g_p: InvertedIndex::new(),
            q_z: InvertedIndex::new(),
            zero_nine: InvertedIndex::new(),
            tokenizer: Tokenizer::new(),
        }
    }

    pub fn add_document(&mut self, doc_id: u16, content: &str) {
        for term in self.tokenizer.tokenize(content) {
            let term = term.to_lowercase();

            let first_char = term.chars().next().unwrap();
            match first_char {
                'a'..='f' => self.a_f.insert(term, doc_id),
                'g'..='p' => self.g_p.insert(term, doc_id),
                'q'..='z' => self.q_z.insert(term, doc_id),
                '0'..='9' => self.zero_nine.insert(term, doc_id),
                _ => {}
            }
        }
    }

    pub fn write_to_disk(&self, location: String) -> std::io::Result<()> {
        // Helper function to convert an InvertedIndex to text format
        fn write_index_to_file(index: &InvertedIndex, path: &str) -> std::io::Result<()> {
            let mut file = File::create(path)?;
            for term in index.get_ordered_keys() {
                if let Some(postings) = index.get_postings(&term) {
                    let postings_str: String = postings
                        .get_postings()
                        .iter()
                        .map(|post| format!("{}|{}", post.doc_id, post.term_freq))
                        .collect::<Vec<_>>()
                        .join(",");

                    writeln!(file, "{}: {}", term, postings_str)?;
                }
            }
            Ok(())
        }

        std::fs::create_dir_all(&location)?;

        write_index_to_file(&self.a_f, &format!("{}/a_f.txt", location))?;
        write_index_to_file(&self.g_p, &format!("{}/g_p.txt", location))?;
        write_index_to_file(&self.q_z, &format!("{}/q_z.txt", location))?;
        write_index_to_file(&self.zero_nine, &format!("{}/0_9.txt", location))?;

        Ok(())
    }
}
