use crate::inverted_index;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::os::windows::fs::MetadataExt;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time;
use url_parse::core::Parser;
const PATH: &str = "../developer/DEV/";
pub const IDBOOK_PATH: &str = "inverted_index/id_book.txt";
pub const BATCH_SIZE: u16 = 5000; // Define the batch size
#[derive(Debug, Deserialize)]

struct Document {
    url: String,
    content: String,
    encoding: String,
}

fn get_only_text_from_html(content: &str, encoding: String) -> String {
    let ascii_content: String = if encoding.to_lowercase().contains("ascii") {
        content.chars().filter(|c| c.is_ascii()).collect::<String>()
    } else {
        match encoding.to_lowercase().as_str() {
            enc if enc.contains("utf-8") => content.to_string(),
            enc if enc.contains("iso-8859") => content.to_string(), // ISO-8859 is already ASCII compatible
            _ => content.chars().filter(|c| c.is_ascii()).collect::<String>(), // fallback to ASCII filtering
        }
    };

    let document = scraper::Html::parse_document(&ascii_content);

    let selector = scraper::Selector::parse("body")
        .unwrap_or_else(|_| scraper::Selector::parse("html").unwrap());

    if let Some(body) = document.select(&selector).next() {
        body.text()
            .collect::<String>()
            .chars()
            .filter(|c| c.is_ascii())
            .collect()
    } else {
        String::new()
    }
}

fn process_file(
    file_path: PathBuf,
    tx_clone: Sender<(u16, String, String)>,
    id_book_clone: Arc<Mutex<HashMap<u16, (String, String)>>>,
    doc_id: Arc<Mutex<u16>>,
) {
    // ! check if the file is valid here
    let file = fs::File::open(&file_path).unwrap();
    if file.metadata().unwrap().file_size() > 5_000_000 {
        return;
    }
    let content: String = fs::read_to_string(&file_path).unwrap();

    let doc: Document = if let Ok(doc) = serde_json::from_str(&content) {
        doc
    } else {
        return;
    };
    let url: String = doc.url.clone();
    if !is_valid_page(&url, &doc.content) {
        return;
    }
    // ! do some logic if there is a query as well perhaps since it could be bad for us
    let text: String = get_only_text_from_html(&doc.content, doc.encoding);
    // * 
    // Send the processed document data to the main thread
    let mut doc_id = doc_id.lock().unwrap();
    *doc_id += 1;
    tx_clone.send((*doc_id, doc.url.clone(), text)).unwrap();
    // Update id_book
    let mut id_book = id_book_clone.lock().unwrap();
    id_book.insert(*doc_id, (doc.url, file_path.to_str().unwrap().to_string()));
}

pub fn main() -> u16 {
    let doc_id: Arc<Mutex<u16>> = Arc::new(Mutex::new(0));
    // shared between threads
    let id_book: Arc<Mutex<HashMap<u16, (String, String)>>> = Arc::new(Mutex::new(HashMap::new()));
    let inverted_indexes: Arc<Mutex<inverted_index::InvertedIndexSplit>> =
        Arc::new(Mutex::new(inverted_index::InvertedIndexSplit::new()));
    let time = time::Instant::now();
    // https://doc.rust-lang.org/book/ch16-02-message-passing.html once you make an index, send it to the main thread to write to disk
    // Create a channel to send data from threads to the main thread
    let (tx, rx): (
        Sender<(u16, String, String)>,
        Receiver<(u16, String, String)>,
    ) = channel();

    let mut handles = vec![]; // Vector to store thread handles
    for dir_entry in fs::read_dir(PATH).unwrap() {
        let dir = dir_entry.unwrap();
        // Iterate over the files in the directory
        // Will error if dir ever contains a non-directory (a file)
        for file in fs::read_dir(dir.path()).unwrap() {
            let file: fs::DirEntry = file.unwrap();
            // assumption is file is a json file
            let filepath = file.path();
            // each sender, needs a way to send, so clone the sender
            let tx_clone = tx.clone();
            // https://doc.rust-lang.org/book/ch16-03-shared-state.html#atomic-reference-counting-with-arct
            // Clone the Arc to share ownership between threads
            let id_book_clone = Arc::clone(&id_book);
            let doc_id_clone = Arc::clone(&doc_id);
            let handle = thread::spawn(move || {
                process_file(filepath, tx_clone, id_book_clone, doc_id_clone);
            });

            handles.push(handle);
        }
    }
    // Drop the original sender to signal the end of sending
    drop(tx);
    // Process documents received from the threads
    let mut batch_count = 0;
    // The loop needs to terminate when all senders are dropped.
    // The `recv()` method returns a `Result`, with `Err` indicating that the channel is closed.
    while let Ok((id, _url, text)) = rx.recv() {
        let mut inverted_indexes_locked = inverted_indexes.lock().unwrap();
        inverted_indexes_locked.add_document(id, &text);
        batch_count += 1;
        // Write to disk if we've processed BATCH_SIZE documents
        if batch_count % BATCH_SIZE as usize == 0 {
            match inverted_indexes_locked.write_to_disk(format!(
                "inverted_index/{}",
                batch_count / BATCH_SIZE as usize - 1
            )) {
                Ok(_) => println!("Successfully written batch to disk:"),
                Err(e) => println!("Error writing to disk: {}", e),
            }
            *inverted_indexes_locked = inverted_index::InvertedIndexSplit::new(); // Reset the index
            println!(
                "Processed {} documents in {} minutes",
                batch_count,
                time.elapsed().as_secs() / 60
            );
        }
    }

    // Complete all threads before continuing to the main thread
    for handle in handles {
        if let Err(e) = handle.join() {
            println!("Error joining thread: {:?}", e);
        }
    }
    let doc_id = *doc_id.lock().unwrap();
    // Write final batch if any documents remain
    let inverted_indexes_locked = inverted_indexes.lock().unwrap();
    if batch_count % BATCH_SIZE as usize != 0 {
        if let Err(e) = inverted_indexes_locked.write_to_disk(format!(
            "inverted_index/{}",
            batch_count / BATCH_SIZE as usize
        )) {
            println!("Error writing final batch to disk: {}", e);
        } else {
            println!(
                "Successfully written final batch to disk, total docs: {}",
                doc_id
            );
        }
    }

    // Write id_book to disk
    let id_book_locked = id_book.lock().unwrap();
    let mut sorted_entries: Vec<_> = id_book_locked.iter().collect();
    sorted_entries.sort_by_key(|&(k, _)| k);

    match fs::File::create(IDBOOK_PATH) {
        Ok(mut file) => {
            for (id, (url, filepath)) in sorted_entries {
                let mut line = format!("{} | {}", url, filepath);
                if line.len() >= 399 {
                    line.truncate(399);
                } else {
                    line.push_str(&" ".repeat(399 - line.len()));
                }
                // making the line exactly 400 characters long for easy random-access reading (writeln! adds a newline)
                if let Err(e) = writeln!(file, "{}", line) {
                    println!("Error writing line for doc_id {}: {}", id, e);
                }
            }
        }
        Err(e) => println!("Error creating id_book file: {}", e),
    }
    return doc_id;
}

fn is_valid_page(url: &str, content: &str) -> bool {
    // Parse URL to validate scheme and check for anchors
    let parsed_url = match Parser::new(None).parse(url) {
        Ok(parsed) => parsed,
        Err(_) => return false,
    };

    // Exclude pages with anchors
    if parsed_url.anchor.is_some() {
        return false;
    }

    // Check file extension
    if let Some(path) = parsed_url.path {
        let path_str = path.last().unwrap_or(&String::new()).to_lowercase();
        let valid_extensions = [
            ".txt", ".html", ".htm", ".md", ".xml", ".xhtml", ".xhtm", ".xht",
        ];

        let has_valid_extension = path_str.is_empty()
            || valid_extensions.iter().any(|ext| path_str.ends_with(ext))
            || !path_str.contains('.');

        if !has_valid_extension {
            return false;
        }
    }

    // Verify content has HTML-like structure
    let content_lower = content.to_lowercase();
    if !content_lower.contains("html")
        && !content_lower.contains("body")
        && !content_lower.contains("meta")
        && !content_lower.contains("<p")
        && !content_lower.contains("txt")
        && !content_lower.contains("text")
    {
        return false;
    }

    true
}
