use std::collections::hash_map::DefaultHasher;
use std::collections::{BinaryHeap, HashMap};
use std::hash::{Hash, Hasher};
use std::ops::BitAnd;

// https://zlib.net/crc_v3.txt
// https://en.wikipedia.org/wiki/Cyclic_redundancy_check polynomial taken from here
const DIVISOR: u64 = 0x42F0E1EBA9EA3693;
pub fn crc32checksum(data: &[u8]) -> u64 {
    let mut crc: u64 = 0x0;

    for byte in data {
        crc = crc ^ (*byte as u64);
        for _ in 0..8 {
            // If the least significant bit is 1
            if crc.bitand(1) == 1 {
                crc = (crc >> 1) ^ DIVISOR;
            } else {
                crc = crc >> 1;
            }
        }
    }
    return crc;
}

// https://wwwconference.org/wp-content/uploads/2025/01/paper215.pdf

// ? Divide the fingerprint into blocks, from these blocks permute them and make a combination of tables :=> these combinations are used to look up the possible actual candidates to probe and chekc with respect to the query finger prints
// ? whats beautiful is checking multiple permutations of the same fingerprint to check for the near duplicates is since you are checking multiple permutations of the same table together, you dont have to recheck the permutaitons of these blocks since you are already doing every possible combination together
// ? (I still have an idea, why make permutations and join that feels memory inefficient, is there a way to simulate it without making the actual permutations while checking this saves memory but increases time both each signifcantly)
// ? Lets test that algorithm here and see if it is possible
pub fn simhash(data: &String) -> u64 {
    let input_str = data.to_lowercase();
    // create features (shingles) - using 3-character shingles
    let features = input_str
        .chars()
        .collect::<Vec<char>>()
        .windows(3)
        .map(|w| w.iter().collect::<String>())
        .collect::<Vec<String>>();
    // initialize the bit counts for 64 bits
    // max you can add is 64 times, aka i8 = 64
    let mut bit_counts = [0i8; 64];
    // hash each feature and update bit counts
    for feature in features {
        let hash = calculate_hash(&feature);
        // update bit counts based on the hash
        for i in 0..64 {
            if (hash & (1 << i)) != 0 {
                bit_counts[i] += 1;
            } else {
                bit_counts[i] -= 1;
            }
        }
    }
    // generate the final hash from bit counts
    let mut result: u64 = 0;
    for i in 0..64 {
        // if its a positive number, set the bit to 1
        if bit_counts[i] > 0 {
            result |= 1 << i;
        }
    }
    result
}

fn calculate_hash(feature: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    feature.hash(&mut hasher);
    hasher.finish()
}

struct SimHashTabler {
    blocks: Vec<SimHashPermTable>,
}

struct SimHashPermTable {
    table: HashMap<u16, BinaryHeap<u16>>,
}

impl SimHashTabler {
    pub fn new() -> SimHashTabler {
        SimHashTabler {
            // For now divide the 64 bit fingerprint into 6 blocks
            blocks: Vec::with_capacity(6),
        }
    }

    pub fn add_permutation(&mut self, fingerprint: u64) {
        // Divide the fingerprint into 6 blocks
        let blocks = fingerprint_to_blocks(&fingerprint);
        // Add the blocks to the table
        for i in 0..6 {
            self.blocks[i]
                .table
                .entry(blocks[i])
                .or_insert(BinaryHeap::new());
            self.blocks[i]
                .table
                .get_mut(&blocks[i])
                .unwrap()
                .push(blocks[i]);
        }
    }

    fn get_candidates(&self, fingerprint: u64) -> Vec<u16> {
        let mut candidates = Vec::new();
        // Divide the fingerprint into 6 blocks
        let blocks = fingerprint_to_blocks(&fingerprint);
        // Get the candidates from the tables
        for i in 0..6 {
            if let Some(table) = self.blocks[i].table.get(&blocks[i]) {
                for candidate in table.iter() {
                    candidates.push(*candidate);
                }
            }
        }

        candidates
    }

    fn is_duplicate(&self, fingerprint: u64) -> bool {
        let mut candidates = self.get_candidates(fingerprint);
        filter_candidates(&mut candidates);
        for candidate in candidates.iter() {
            if hamming_distance(fingerprint, *candidate as u64) < 3 {
                return true;
            }
        }
        false
    }
}
fn hamming_distance(fingerprint1: u64, fingerprint2: u64) -> u8 {
    let xor_result = fingerprint1 ^ fingerprint2;
    xor_result.count_ones() as u8
}
fn fingerprint_to_blocks(fingerprint: &u64) -> Vec<u16> {
    let mut blocks = Vec::with_capacity(6);
    for i in 0..6 {
        blocks.push((fingerprint >> (i * 10)) as u16);
    }
    blocks
}

fn filter_candidates(candidates: &mut Vec<u16>) {
    // if the candidate exists more than 3 times, then keep it
    let mut counts = HashMap::new();
    for &candidate in candidates.iter() {
        *counts.entry(candidate).or_insert(0) += 1;
    }
    candidates.retain(|&candidate| counts.get(&candidate).map_or(false, |&count| count > 3));
}
