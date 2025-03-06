use std::{
    fs::File,
    io::{BufRead, BufReader, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

use crate::postings::Postings;

pub const MERGED_INDEX_DIR: &str = "inverted_index/merged";
const PARTITION: u16 = 100;

pub struct FileSkip {
    character: char,
    word: String,
    byte_offset: u64,
}

type FileSkipList = Vec<FileSkip>;

impl FileSkip {
    pub fn build_skip_list(path: PathBuf) -> FileSkipList {
        let initial_character: char = path
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap()
            .chars()
            .next()
            .unwrap();

        // if file doesn't exist, create it
        let file: File = if path.exists() {
            File::open(&path).unwrap()
        } else {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            File::create(&path).unwrap()
        };
        // create a buffered reader, which is more efficient for reading lines
        let mut reader: BufReader<File> = BufReader::new(file);
        // * from the documentation:
        // BufReader<R> can improve the speed of programs that make small and
        // repeated read calls to the same file.
        // It does not help when reading very large amounts at once, or reading just one or a few times.
        // It also provides no advantage when reading from a source that is already in memory, like a Vec<u8>.

        let mut skip_list: FileSkipList = Vec::new();
        let mut line_count: u32 = 0;
        let mut line: String = String::new();
        // read_line appends to the provided buffer, gets the line from the buffer and returns the number of bytes read
        while let Ok(bytes_read) = reader.read_line(&mut line) {
            if bytes_read == 0 {
                break;
            }

            // Get current position using the reader's position
            // self.seek(SeekFrom::Current(0)) equivalent to self.stream_position()
            // stream_position will point to the end of this line, so we need to subtract the bytes_read to get the start of the line
            let current_position = reader.stream_position().unwrap() - bytes_read as u64;

            // for every partition-th line, add a skip entry!!
            if line_count % PARTITION as u32 == 0 {
                if let Some(word) = line.split(':').next() {
                    skip_list.push(FileSkip {
                        character: initial_character,
                        word: word.to_string(),
                        byte_offset: current_position,
                    });
                }
            }

            line_count += 1;
            line.clear();
        }

        skip_list
    }

    pub fn write_skip_list(skip_list: &FileSkipList) {
        if skip_list.is_empty() {
            return;
        }

        let path = Path::new(MERGED_INDEX_DIR);
        if !path.exists() {
            std::fs::create_dir_all(path).unwrap();
        }

        let character = skip_list[0].character;
        let file_path = format!("{}/{}_skiplist.txt", MERGED_INDEX_DIR, character);
        let mut file = File::create(file_path).unwrap();

        // Write skip list entries to file
        for (i, skip) in skip_list.iter().enumerate() {
            let entry = if i == skip_list.len() - 1 {
                format!("{}:{}", skip.word.trim(), skip.byte_offset)
            } else {
                format!("{}:{},", skip.word.trim(), skip.byte_offset)
            };
            file.write_all(entry.as_bytes()).unwrap();
        }
    }

    pub fn read_skip_list(character: char) -> FileSkipList {
        let file_path = format!("{}/{}_skiplist.txt", MERGED_INDEX_DIR, character);
        let path = Path::new(&file_path);

        if !path.exists() {
            return Vec::new();
        }

        let mut file = File::open(path).unwrap();
        let mut read_buffer = String::new();
        file.read_to_string(&mut read_buffer).unwrap();

        let mut skip_list = Vec::new();
        for entry in read_buffer.split(',') {
            if entry.is_empty() {
                continue;
            }

            if let Some((word, offset_str)) = entry.split_once(':') {
                if let Ok(byte_offset) = offset_str.parse::<u64>() {
                    skip_list.push(FileSkip {
                        character,
                        word: word.to_string(),
                        byte_offset,
                    });
                }
            }
        }

        skip_list
    }

    // ? Could clean this up and even use binary search in finding the correct skip list
    pub fn find_skip_entry(skip_list: &FileSkipList, word: &String) -> WordOffsetRange {
        if skip_list.is_empty() || word < &skip_list[0].word {
            return WordOffsetRange::Invalid;
        }

        let mut prev_offset = 0;

        for i in 0..skip_list.len() {
            let skip = &skip_list[i];

            if &skip.word == word {
                return WordOffsetRange::Exact(skip.byte_offset);
            }

            if &skip.word < word {
                prev_offset = skip.byte_offset;
                continue;
            } else if &skip.word > word {
                if i == 0 {
                    return WordOffsetRange::Invalid;
                }
                return WordOffsetRange::Between(prev_offset, skip.byte_offset);
            }

            prev_offset = skip.byte_offset;
        }

        // the word must be after the last entry
        println!("Word is after the last entry in the skip list");
        WordOffsetRange::After(prev_offset)
    }
}
pub enum WordOffsetRange {
    Invalid,
    Exact(u64),        // Word is an exact match
    Between(u64, u64), // Word is between two offsets
    After(u64),        // Word is after the last entry in the skip list
}

pub fn get_postings_from_offset_range(
    file: &File,
    offset_range: WordOffsetRange,
    word: &str,
) -> Postings {
    let mut reader = BufReader::new(file);
    let mut postings = Vec::new();

    match offset_range {
        // if we found the exact word, read the line and load the postings
        WordOffsetRange::Exact(offset) => {
            reader.seek(SeekFrom::Start(offset)).unwrap();
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            postings = Postings::load_postings(&line).unwrap().postings;
        }
        WordOffsetRange::Between(start_offset, end_offset) => {
            // read from start_offset to end_offset and load the postings
            reader.seek(SeekFrom::Start(start_offset)).unwrap();
            let mut line = String::new();
            // read line by line until we reach the end_offset
            while reader.stream_position().unwrap() < end_offset {
                reader.read_line(&mut line).unwrap();
                if let Ok(posting) = Postings::load_postings(&line) {
                    if posting.word == word {
                        postings.extend(posting.postings);
                    }
                }
                line.clear();
            }
        }
        // if the word is after the last entry in the skip list, read from the offset to the end of the file
        WordOffsetRange::After(offset) => {
            reader.seek(SeekFrom::Start(offset)).unwrap();
            let mut line = String::new();
            while reader.read_line(&mut line).unwrap() > 0 {
                if let Ok(posting) = Postings::load_postings(&line) {
                    if posting.word == word {
                        postings.extend(posting.postings);
                    }
                }
                line.clear();
            }
        }
        _ => {}
    }

    Postings {
        word: word.to_string(),
        postings,
        skip_list: Vec::new(),
    }
}
