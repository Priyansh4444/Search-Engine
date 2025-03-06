// ! Not a file in use, mainly for testing, debugging and looking at statistics
use std::fs::{self, File};
use std::os::windows::fs::MetadataExt;
use std::sync::{Arc, Mutex};
use std::thread;
pub fn main() {
    let twty = Arc::new(Mutex::new(Vec::new()));
    let five = Arc::new(Mutex::new(Vec::new()));
    let files = get_files_in_dir("../developer/DEV");

    let chunk_size = 5000;
    let mut handles = vec![];

    for chunk in files.chunks(chunk_size) {
        let chunk_files = chunk.to_vec();
        let twty_clone = Arc::clone(&twty);
        let five_clone = Arc::clone(&five);

        let handle: thread::JoinHandle<()> = thread::spawn(move || {
            for file in chunk_files {
                match File::open(&file) {
                    Ok(file_string) => {
                        if let Ok(metadata) = file_string.metadata() {
                            let size = metadata.file_size();
                            if size > 20000000 {
                                twty_clone.lock().unwrap().push(file);
                            } else if size > 5000000 {
                                five_clone.lock().unwrap().push(file);
                            }
                        }
                    }
                    Err(_) => continue,
                }
            }
        });

        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Files with size > 20MB: {:?}", *twty.lock().unwrap());
    println!("Files with size > 5MB: {:?}", *five.lock().unwrap());
}

fn get_files_in_dir(dir: &str) -> Vec<String> {
    let mut files = Vec::new();
    let paths = fs::read_dir(dir).unwrap();

    for path in paths {
        let path = path.unwrap().path();
        let path_str = path.to_str().unwrap().to_string();
        if path.is_dir() {
            files.append(&mut get_files_in_dir(&path_str));
        } else {
            files.push(path_str);
        }
    }

    files
}
