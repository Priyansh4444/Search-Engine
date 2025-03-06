pub mod file_skip_list;
pub mod id_book;
pub mod index_builder;
pub mod inverted_index;
pub mod lazy_merger;
pub mod postings;
pub mod query;
pub mod single_posting;
pub mod tokenizer;
use query::SearchEngine;

fn main() {
    // ! BUILD INDEX
    // let doc_id = index_builder::main();
    // ! MERGE BATCHES
    // ! The following code snippet merges the batches of inverted indexes into a multiple sorted inverted index.
    // lazy_merger::main(doc_id);

    // ! Comes searching and ranking now

    println!("Welcome to the Search Engine!");
    let mut search_engine: SearchEngine = SearchEngine::new();
    loop {
        search_engine.get_query();
        search_engine.search();
    }
}

// How are we linking to the dataset files though, usually i would at least link it via the word "here" and then the link to the file
