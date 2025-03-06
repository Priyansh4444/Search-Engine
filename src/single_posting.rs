#[derive(Debug, Clone)]
pub struct Posting {
    pub doc_id: u16,
    pub term_freq: u16,
}

impl Ord for Posting {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.doc_id.cmp(&other.doc_id)
    }
}

impl PartialOrd for Posting {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for Posting {
    // This is the only implementation that is correct.
}

impl PartialEq for Posting {
    fn eq(&self, other: &Self) -> bool {
        self.doc_id == other.doc_id
    }
}

#[allow(dead_code)]
impl Posting {
    pub fn new(doc_id: u16, term_freq: u16) -> Posting {
        Posting { doc_id, term_freq }
    }

    pub fn increment_freq(&mut self) {
        self.term_freq += 1;
    }
}
