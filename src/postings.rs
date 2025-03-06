use crate::single_posting::Posting;

#[derive(Clone)]
pub struct Postings {
    pub word: String,
    pub postings: Vec<Posting>,
    pub skip_list: Vec<SkipList>,
}

#[derive(Clone)]
pub struct SkipList {
    doc_id: u16,
    index: u16,
}

impl Postings {
    pub fn new(term: String) -> Postings {
        Postings {
            word: term,
            postings: Vec::new(),
            skip_list: Vec::new(),
        }
    }

    pub fn push(&mut self, posting: Posting) {
        // Find the position where the new posting should be inserted
        let insert_pos = self
            .postings
            .binary_search_by_key(&posting.doc_id, |p| p.doc_id)
            .unwrap_or_else(|pos| pos);
        // Insert the posting at the correct position
        self.postings.insert(insert_pos, posting);
    }

    pub fn update_frequency(&mut self, doc_id: u16) {
        let pos = self.postings.binary_search_by_key(&doc_id, |p| p.doc_id);
        match pos {
            Ok(pos) => {
                self.postings[pos].increment_freq();
            }
            Err(_) => {
                self.push(Posting::new(doc_id, 1));
            }
        }
    }

    pub fn get_postings(&self) -> &Vec<Posting> {
        &self.postings
    }

    pub fn load_postings(line: &str) -> Result<Postings, &'static str> {
        if line.is_empty() {
            return Err("Empty line");
        }
        let (word, postings_str) = line.split_once(':').unwrap();
        let mut postings = Postings::new(word.to_string());
        for single_posting in postings_str.split(",") {
            let (doc_id, term_frequency) = single_posting.split_once("|").unwrap();
            postings.push(Posting::new(
                doc_id.trim().parse::<u16>().unwrap(),
                term_frequency.trim().parse::<u16>().unwrap(),
            ));
        }
        return Ok(postings);
    }

    pub fn merge(&mut self, other: Postings) {
        if self.word != other.word {
            panic!("Merging two different terms");
        }
        let mut i = 0;
        let mut j = 0;
        while i < self.postings.len() && j < other.postings.len() {
            if self.postings[i].doc_id < other.postings[j].doc_id {
                i += 1;
            } else if self.postings[i].doc_id > other.postings[j].doc_id {
                self.postings.insert(i, other.postings[j].clone());
                i += 1;
                j += 1;
            }
        }
        while j < other.postings.len() {
            self.postings.push(other.postings[j].clone());
            j += 1;
        }
    }

    pub fn save_postings(&self) -> String {
        let mut result = String::new();
        result.push_str(&self.word);
        result.push(':');
        for posting in &self.postings {
            result.push_str(&format!("{}|{},", posting.doc_id, posting.term_freq));
        }
        result.pop();
        result
    }

    // ? Unused for now since if you are going to load the entire postlist into memory, you dont need skip list
    pub fn build_skip_list(&mut self) {
        let mut skip_list = Vec::with_capacity(5);
        let postings_length = self.postings.len();
        // Create a skip list with 5 entries
        if postings_length >= 5 {
            let step = postings_length / 5;
            for i in 0..5 {
                skip_list.push(SkipList {
                    doc_id: self.postings[i * step].doc_id,
                    index: (i * step) as u16,
                });
            }
        } else {
            // If less than 5 elements, just add what we have
            for i in 0..postings_length {
                skip_list.push(SkipList {
                    doc_id: self.postings[i].doc_id,
                    index: i as u16,
                });
            }
        }
        self.skip_list = skip_list;
    }

    // ? Unused for now since if you are going to load the entire postlist into memory, you dont need skip list
    pub fn skip_list_to_file(&self) -> String {
        let result = self
            .skip_list
            .iter()
            .map(|x| format!("{}|{}", x.doc_id, x.index))
            .collect::<Vec<String>>()
            .join(",");

        result
    }
}