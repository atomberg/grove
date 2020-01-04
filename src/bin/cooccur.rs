extern crate getopts;
extern crate grove;

#[macro_use]
extern crate log;

use env_logger;
use getopts::Options;
use std::collections::{BinaryHeap, HashMap};
use std::env;
use std::fs;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use twox_hash::RandomXxHashBuilder64;

use grove::{estimate_max_prod, record_io::Records, sparse::CornerMatrix, sparse::SparseRecord};

struct Params {
    window_size: usize,
    symmetric: bool,
    memory_limit: f32,
    max_product: usize,
    vocab_file: String,
    file_head: String,
    overflow_length: usize,
    verbose: usize,
}

#[derive(PartialEq, PartialOrd, Eq, Ord)]
struct HeapElement {
    record: SparseRecord<f32>,
    file_id: usize,
}

fn main() {
    env_logger::init();
    let params = parse_args();

    info!(
        "Parsed arguments: max_product={}, overflow_length={}",
        params.max_product, params.overflow_length
    );

    let vocab = read_vocab(params.vocab_file);
    info!("Read {} tokens into the vocabulary hash map", vocab.len());
    debug!("{:?}", vocab);

    let mut bigram_table = CornerMatrix::<f32>::new(vocab.len(), params.max_product);
    let mut overflow_buffer = Vec::<SparseRecord<f32>>::with_capacity(params.overflow_length);

    let mut reader = BufReader::new(io::stdin());
    let mut line = String::new();
    let mut words = Vec::<usize>::with_capacity(32);
    let mut counter: usize = 0;
    let mut tmp_files = Vec::<String>::new();
    while reader.read_line(&mut line).unwrap() > 0 {
        debug!("Line: '{}'", line);
        for word in line.split_whitespace().map(|word| vocab.get(word)) {
            counter += 1;
            if let Some(w) = word {
                words.push(*w - 1);
                // } else {
                //     // Deal with unk case?
                //     panic!("A word is not in the vocabulary");
            }
        }
        debug!("Words: '{:?}'", words);
        if overflow_buffer.capacity() < overflow_buffer.len() + words.len() {
            // Flush the overflow buffer and truncate it
            tmp_files.push(format!("{}_{:04}.bin", params.file_head, tmp_files.len()));
            flush_overflow_buffer(&mut overflow_buffer, tmp_files.last().unwrap().clone());
            overflow_buffer.truncate(0);
        }
        for (focus_idx, &focus_rank) in words.iter().enumerate() {
            debug!("Focus: idx={}, rank={}", focus_idx, focus_rank);
            // Using saturating_sub to get a lower bound of zero without any extra operations
            for (context_idx, &context_rank) in words[focus_idx.saturating_sub(params.window_size)..focus_idx]
                .iter()
                .enumerate()
            {
                let cooc = 1.0 / (focus_idx as f32 - context_idx as f32) as f32;
                debug!("Processing ({}, {}, {})", focus_rank, context_rank, cooc);
                if (focus_rank + 1) * (context_rank + 1) <= bigram_table.max_prod {
                    *bigram_table.get(focus_rank, context_rank) += cooc;
                } else {
                    overflow_buffer.push(SparseRecord {
                        row: focus_rank as u32,
                        col: context_rank as u32,
                        val: cooc,
                    });
                }
            }
        }
        words.clear();
        line.clear();
    }
    // Flush the overflow buffer one last time
    if !overflow_buffer.is_empty() {
        tmp_files.push(format!("{}_{:04}.bin", params.file_head, tmp_files.len()));
        flush_overflow_buffer(&mut overflow_buffer, tmp_files.last().unwrap().clone());
    }
    info!("Processed {} tokens", counter);
    debug!("Table = {:?}", bigram_table);
    debug!("Overflow = {:?}", overflow_buffer);

    let mut writer = BufWriter::new(io::stdout());
    for record in bigram_table.to_sparse() {
        match writer.write_all(&record.to_bytes()) {
            Ok(n) => n,
            Err(e) => panic!("Could not write to stdout: {}", e.to_string()),
        };
    }

    info!("Wrote {} dense elements of the cooccurrence matrix", bigram_table.len());

    if !tmp_files.is_empty() {
        merge_temp_files(tmp_files, &mut writer);
    }
}

fn merge_temp_files(tmp_files: Vec<String>, writer: &mut dyn Write) {
    let mut pq: BinaryHeap<HeapElement> = BinaryHeap::new();
    let mut file_readers: Vec<Records<std::io::BufReader<std::fs::File>>> = Vec::with_capacity(tmp_files.len());
    for (file_id, file) in tmp_files.iter().enumerate() {
        let mut reader = Records {
            buffer: [0; 12],
            filename: file.to_string(),
            reader: BufReader::new(match fs::File::open(&file) {
                Ok(file) => file,
                Err(e) => panic!("Could not open {}: {}", file, e.to_string()),
            }),
        };
        if let Some(result) = reader.next() {
            if let Ok(record) = result {
                pq.push(HeapElement { record, file_id });
            }
        } else {
            panic!(
                "Could not read 20 bytes from {} and parse them into a SparseRecord",
                &reader.filename
            );
        }
        file_readers.push(reader);
    }

    // Pop from the heap
    let mut cur: HeapElement = pq.pop().unwrap();
    let mut prev: HeapElement;

    debug!("Current = {}, {:?}, Prev = None", cur.file_id, cur.record);

    // Push onto the heap from the same file as the popped element
    match file_readers[cur.file_id].next() {
        Some(result) => match result {
            Ok(record) => pq.push(HeapElement {
                record,
                file_id: cur.file_id,
            }),
            Err(e) => panic!(
                "Could not read 20 bytes from {}: {}",
                file_readers[cur.file_id].filename,
                e.to_string()
            ),
        },
        None => {
            // &file_readers[cur.file_id].reader.close();
            panic!(
                "Could not parse bytes {:?} into a SparseRecord",
                &file_readers[cur.file_id].buffer
            )
        }
    };

    while !pq.is_empty() {
        prev = cur;
        cur = pq.pop().unwrap();

        debug!(
            "Current = {}, {:?}, Prev = {}, {:?}",
            cur.file_id, cur.record, prev.file_id, prev.record
        );

        if prev.record == cur.record {
            cur.record.val += prev.record.val;
        } else {
            match writer.write_all(&prev.record.to_bytes()) {
                Ok(n) => n,
                Err(e) => panic!("Could not write: {}", e.to_string()),
            };
        }

        // Push onto the heap from the same file as the popped element
        if let Some(result) = file_readers[cur.file_id].next() {
            if let Ok(record) = result {
                pq.push(HeapElement {
                    record,
                    file_id: cur.file_id,
                })
            } else {
                info!("File {} reached EOF", file_readers[cur.file_id].filename);
            }
        }
    }

    // Heap is empty now, but cur was not flushed to stdout yet
    match writer.write_all(&cur.record.to_bytes()) {
        Ok(n) => n,
        Err(e) => panic!("Could not write: {}", e.to_string()),
    };
}

fn read_vocab(vocab_filename: String) -> HashMap<String, usize, RandomXxHashBuilder64> {
    let mut reader = BufReader::new(match fs::File::open(&vocab_filename) {
        Ok(file) => file,
        Err(e) => panic!("Could not open {}: {}", vocab_filename, e.to_string()),
    });

    let mut map: HashMap<String, usize, RandomXxHashBuilder64> = Default::default();
    let mut line = String::new();
    let mut rank = 1;
    while reader.read_line(&mut line).unwrap() > 0 {
        let words: Vec<&str> = line.split_whitespace().collect();
        map.insert(words[0].to_string(), rank);
        line.clear();
        rank += 1;
    }
    map
}

fn flush_overflow_buffer(overflow_buffer: &mut Vec<SparseRecord<f32>>, filename: String) {
    overflow_buffer.sort_unstable();
    let mut writer = BufWriter::new(match fs::File::create(filename.clone()) {
        Ok(file) => file,
        Err(e) => panic!("Could not create {}: {}", filename, e.to_string()),
    });
    let mut cur_rec = match overflow_buffer.get(0) {
        Some(&rec) => rec,
        None => {
            return;
        }
    };
    for rec in overflow_buffer[1..].iter() {
        if cur_rec == *rec {
            cur_rec.val += rec.val
        } else {
            match writer.write_all(&cur_rec.to_bytes()) {
                Ok(n) => n,
                Err(e) => panic!("Could write to {}: {}", filename, e.to_string()),
            };
            cur_rec = SparseRecord {
                row: rec.row,
                col: rec.col,
                val: rec.val,
            };
        }
    }
    match writer.write_all(&cur_rec.to_bytes()) {
        Ok(n) => n,
        Err(e) => panic!("Could not write to {}: {}", filename, e.to_string()),
    };
}

fn parse_args() -> Params {
    let mut params = Params {
        window_size: 15,
        symmetric: false,
        memory_limit: 4.0,
        max_product: 0,
        vocab_file: "vocab.txt".to_string(),
        file_head: "overflow".to_string(),
        overflow_length: 0,
        verbose: 2,
    };

    let args: Vec<String> = env::args().collect();

    let mut opts = Options::new();
    opts.optopt("", "verbose", "verbosity level (default 2)", "<int>");
    opts.optflag(
        "",
        "symmetric",
        "if present, use left and right contexts, if not, use left context only",
    );
    opts.optopt(
        "",
        "window-size",
        "Number of context words to the left (and to the right, if symmetric=1); default 15",
        "<int>",
    );
    opts.optopt(
        "",
        "vocab-file",
        "File containing vocabulary (truncated unigram counts, produced by 'vocab_count'); default vocab.txt",
        "<file>",
    );
    opts.optopt(
        "",
        "memory-limit",
        "Soft limit for memory consumption, in GB -- based on simple heuristic, so not extremely accurate; default 4.0",
        "<float>",
    );
    opts.optopt(
        "",
        "max-product",
        "Limit the size of dense cooccurrence array by specifying the max product <int> of \
         the frequency counts of the two cooccurring words. This value overrides that \
         which is automatically produced by '--memory-limit'. \
         Typically only needs adjustment for use with very large corpora.",
        "<int>",
    );
    opts.optopt(
        "",
        "overflow-file",
        "Filename, excluding extension, for temporary file; default overflow",
        "<file>",
    );
    opts.optflag("h", "help", "print this help menu");
    let matches = match opts.parse(&args[1..]) {
        Ok(m) => m,
        Err(f) => panic!(f.to_string()),
    };
    if matches.opt_present("help") {
        print!(
            "{}",
            opts.usage("Usage: ./cooccur [options] < corpus.txt > cooccurrences.bin")
        );
        std::process::exit(0);
    } else {
        params.verbose = match matches.opt_get_default("verbose", params.verbose) {
            Ok(m) => m,
            Err(f) => panic!(f.to_string()),
        };
        params.symmetric = matches.opt_present("symmetric");
        params.window_size = match matches.opt_get_default("window-size", params.window_size) {
            Ok(m) => m,
            Err(f) => panic!(f.to_string()),
        };
        params.vocab_file = match matches.opt_get_default("vocab-file", params.vocab_file) {
            Ok(m) => m,
            Err(f) => panic!(f.to_string()),
        };
        params.memory_limit = match matches.opt_get_default("memory-limit", params.memory_limit) {
            Ok(m) => m,
            Err(f) => panic!(f.to_string()),
        };
        params.max_product = match matches.opt_get_default("max-product", params.max_product) {
            Ok(m) => {
                debug!("Setting max_prod={}", m);
                m
            }
            Err(f) => panic!(f.to_string()),
        };
        params.file_head = match matches.opt_get_default("overflow-file", params.file_head) {
            Ok(m) => m,
            Err(f) => panic!(f.to_string()),
        };
    }
    if params.max_product == 0 {
        let n = params.memory_limit * (2.0 as f32).powi(30) / 120.0 * 5.0;
        params.max_product = estimate_max_prod(n, 1e-3);
        debug!(
            "Since max_prod was 0, we set it to a heuristic value of {} (N = {})",
            params.max_product, n
        );
    }
    if params.overflow_length == 0 {
        params.overflow_length = (params.memory_limit * (2.0 as f32).powi(30) / 120.0) as usize;
    }
    params
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab() {
        let n = estimate_max_prod(1024.0, 1e-3);
        assert_eq!(n, 189);
    }
}
