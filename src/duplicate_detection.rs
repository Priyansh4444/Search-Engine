use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
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
// section three mentions how you should sort these and store these instead of
// checking all of the hashes that exist since thats time expensive and rather storing every single hash in the table since that would be memory expensive

// since we only want to see a difference of three we store them in a sorted order and compare
// we can quickly search only the nodes we need to check that could be different, like is there a number in which this bit is different or these two bits are different or these three bits are different
// this would not scale with a threshold of 4 or 5 since the number of comparisons would be too high, but we are taking advantage of the fact that hashing is compressing the data into a smaller vector making it easier to compare hashes so it might be viable
// Only check the remaining (less significant) bits of the fingerprints that matched in the first step.
// To make sure you catch every possible match, you might need to create a few extra copies of your sorted list, each sorted slightly differently. This helps you cover all the possible locations of the bit differences.

// * Thought, what if we used a binary tree and as soon as we break three steps, its a different hash,
// * but how do i store a binary tree in disk since we are going to have a lot of hashes

// for k = 3 bits, we have to search for 2^(d-p(i))

// * this paper mentions interpolation search which is not a search im used to but seems to be log(log(n)) which is MUCH faster than binary search

// since we would want to parallelize checking if simhashes were similar what they do is dividde u64 into 4 parts and check if each part is similar with the 4 parts having their own tables
// each division of 4 is 16 bits each, and they each have  a table with 2^16 entries since there are those many permutations of 16 bits, this uniformity is also what gives it its speed with interpolation search
// the more the number of tables we see that the less amount of queries we have to make, since the permutations becomes significantly less!!!
// making it super fast with 6 tables but that would take ALOT more space!, with map reduce also this could be parallelized which is a huge advantage with map reduce!

// since I dont have mapreduce capabilities. currently I will be doing it the brute force way, and hopefully down the line I can implement the interpolation search with at least two tables
// each table just proposes candidates to check, and then we do a hamming distance check to see if they are similar or not
// if any of the candidates differ by more than 3 bits during choosing the candidates, then we can safely say that they are not similar
pub fn simhash(data: &String) -> u64 {
    let input_str = data.to_lowercase();
    // create features (shingles) - using 3-character shingles
    let features = make_features(&input_str);
    // initialize the bit counts for 64 bits - i8 is sufficient for counting
    let mut bit_counts = [0u8; 64];
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
        if bit_counts[i] > 0 {
            result |= 1 << i;
        }
    }

    result
}

fn make_features(input_str: &str) -> Vec<String> {
    // K = 3 as mentioned is a good length from the paper
    let length = 3;
    let cleaned: String = input_str.chars().filter(|c| c.is_alphanumeric()).collect();
    // If string is too short, return the whole string
    if cleaned.len() <= length {
        return vec![cleaned];
    }
    // create shingles
    let mut features = Vec::new();
    for i in 0..=(cleaned.len() - length) {
        features.push(cleaned[i..(i + length)].to_string());
    }
    features
}

fn calculate_hash(feature: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    feature.hash(&mut hasher);
    hasher.finish()
}

// * first u16 point to the bits which are the most significant, and the Vec<u16> are the doc_ids that would be candidates if they were to be matched
struct SimHashes {
    first_16: HashMap<u16, Vec<u16>>,
    second_16: HashMap<u16, Vec<u16>>,
    third_16: HashMap<u16, Vec<u16>>,
    fourth_16: HashMap<u16, Vec<u16>>,
}
// * things to do and optimize this further would be to sort and store the keys I think since there would be a lot of keys in the hashmap which are supposed to be offloaded to disk
// * the biggest problem is that 16 bits can hold 65,536 values each, and we are effectively storing a lot of docIds (each 16 bytes) as well, which is not feasible in memory especially as a vec, at the scale of 
// * 8billion that would average 4 tables × billions of entries
// sorting would slow down search here, but sorting can fix this with interpolation search with a candidate based apporach reducing candidates to check

