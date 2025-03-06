use crate::index_builder::BATCH_SIZE;
use crate::postings::Postings;
use std::fs;
use std::io::{BufRead, BufReader, LineWriter, Write};
use std::path::PathBuf;

use crate::file_skip_list::{FileSkip, MERGED_INDEX_DIR};
pub fn main(doc_id: u16) {
    fs::create_dir_all(MERGED_INDEX_DIR).unwrap_or_default();
    let batch_count = (doc_id - 1) / BATCH_SIZE + 1;
    let word_ranges = ["0_9", "a_f", "g_p", "q_z"];
    // first character is null
    let mut current_first_char = '\0';
    let mut final_file_appender: Option<LineWriter<fs::File>> = None;

    for &words in &word_ranges {
        // Open all available files for this word range
        let mut readers = Vec::with_capacity(batch_count as usize);

        // Open all available files for this word range
        for i in 0..batch_count {
            let filepath = format!("inverted_index/{}/{}.txt", i, words);
            if let Ok(file) = fs::File::open(&filepath) {
                readers.push((BufReader::new(file), String::new()));
            }
        }

        if readers.is_empty() {
            continue;
        }

        // initialize every single posting with a line from the respective files, if available else remove it
        let mut postings_with_indices = Vec::with_capacity(readers.len());
        for (i, (reader, _)) in readers.iter_mut().enumerate() {
            let mut line = String::new();
            if reader.read_line(&mut line).unwrap_or(0) > 0 && !line.trim().is_empty() {
                if let Ok(posting) = Postings::load_postings(&line) {
                    postings_with_indices.push((posting, i));
                }
            }
        }

        while !postings_with_indices.is_empty() {
            // find smallest word
            postings_with_indices.sort_unstable_by(|a, b| a.0.word.cmp(&b.0.word));
            let smallest_word = postings_with_indices[0].0.word.clone();

            // first character check - create new file if needed
            let first_char = smallest_word.chars().next().unwrap_or('0');
            if first_char != current_first_char {
                if let Some(writer) = final_file_appender.as_mut() {
                    writer.flush().unwrap();
                }
                // Build and write skip list for previous character unless its starting character of null
                if current_first_char != '\0' {
                    // build a skip list on that file
                    let skip_list_path =
                        PathBuf::from(format!("{}/{}.txt", MERGED_INDEX_DIR, current_first_char));
                    let file_skip_list = FileSkip::build_skip_list(skip_list_path);
                    FileSkip::write_skip_list(&file_skip_list);
                }
                // append the postings to the new file
                current_first_char = first_char;
                let file_path = format!("{}/{}.txt", MERGED_INDEX_DIR, current_first_char);
                final_file_appender = Some(LineWriter::new(fs::File::create(file_path).unwrap()));
            }

            // merge all postings with the smallest word
            let mut merged_posting = Postings::new(smallest_word);
            let mut indices_to_update = Vec::new();

            let mut i = 0;
            // merge all postings with the smallest word
            while i < postings_with_indices.len() {
                if postings_with_indices[i].0.word == merged_posting.word {
                    merged_posting.merge(postings_with_indices[i].0.clone());
                    indices_to_update.push(postings_with_indices[i].1);
                    postings_with_indices.remove(i);
                } else {
                    i += 1;
                }
            }

            // Write merged posting
            if let Some(writer) = final_file_appender.as_mut() {
                writer
                    .write_all((merged_posting.save_postings() + "\n").as_bytes())
                    .unwrap();
            }

            // Read next lines for updated readers
            for &idx in &indices_to_update {
                let mut line = String::new();
                if readers[idx].0.read_line(&mut line).unwrap_or(0) > 0 && !line.trim().is_empty() {
                    if let Ok(posting) = Postings::load_postings(&line) {
                        postings_with_indices.push((posting, idx));
                    }
                }
            }
        }
    }

    // Flush and create skip list for the last file
    if let Some(writer) = final_file_appender.as_mut() {
        writer.flush().unwrap();
    }

    if current_first_char != '\0' {
        let skip_list_path =
            PathBuf::from(format!("{}/{}.txt", MERGED_INDEX_DIR, current_first_char));
        let file_skip_list = FileSkip::build_skip_list(skip_list_path);
        FileSkip::write_skip_list(&file_skip_list);
    }
}
