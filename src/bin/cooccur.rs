extern crate getopts;
extern crate grove;

use getopts::Options;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use twox_hash::RandomXxHashBuilder64;

use grove::{estimate_max_prod, CornerMatrix, SparseRecord};

struct CooccurRecordWithId(u32, u32, f32, u32);

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

fn main() {
    let params = parse_args();
    let vocab_hash = read_vocab(params.vocab_file);

    let mut writer = BufWriter::new(io::stdout());

    let n = 189;
    let table = CornerMatrix::<f32>::new(5, 10, 0.0);

    let overflow_buffer = Vec::<SparseRecord>::with_capacity(params.overflow_length);
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

fn process_line(
    vocab: &HashMap<String, usize, RandomXxHashBuilder64>,
    bigram_table: &mut CornerMatrix<f32>,
    overflow_buffer: &mut Vec<SparseRecord>,
) {
    let mut reader = BufReader::new(io::stdin());

    let mut line = String::new();
    while reader.read_line(&mut line).unwrap() > 0 {
        let words: Vec<usize> = line
            .split_whitespace()
            .collect()
            .map(|word| vocab.get(word));
        if overflow_buffer.capacity() < overflow_buffer.len() + words.len() {
            // Flush the overflow buffer and truncate it
            flush_overflow_buffer(&mut overflow_buffer, "whatever".to_string());
            overflow_buffer.truncate(0);
        }
        for (i, &focus_word) in words.iter().enumerate() {
            for (j, &context_word) in words.iter().enumerate() {
                let cooc = 1.0 / (i - j) as f32;
                if i * j < bigram_table.max_prod {
                    *bigram_table.get(focus_word, context_word) += cooc;
                } else {
                    overflow_buffer.push(SparseRecord {
                        w1: focus_word,
                        w2: context_word,
                        cooc,
                    });
                }
            }
        }
        line.clear();
    }
}

fn flush_overflow_buffer(overflow_buffer: &mut Vec<SparseRecord>, filename: String) {
    overflow_buffer.sort_unstable_by(|lhs, rhs| match lhs.w1.cmp(&rhs.w1) {
        Ordering::Equal => lhs.w2.cmp(&rhs.w2),
        x => x,
    });
    let mut writer = BufWriter::new(match fs::File::create(filename) {
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
        if rec.w1 == cur_rec.w1 && rec.w2 == cur_rec.w2 {
            cur_rec.cooc += rec.cooc
        } else {
            writer.write(&cur_rec.to_bytes());
            cur_rec = SparseRecord {
                w1: rec.w1,
                w2: rec.w2,
                cooc: rec.cooc,
            };
        }
    }
    writer.write(&cur_rec.to_bytes());
}

fn parse_args() -> Params {
    let mut params = Params {
        window_size: 15,
        symmetric: true,
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
    opts.optopt(
        "",
        "symmetric",
        "if false, only use left context; if true, use left and right (default true)",
        "<bool>",
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
        "<file>"
    );
    opts.optopt(
        "",
        "memory-limit",
        "Soft limit for memory consumption, in GB -- based on simple heuristic, so not extremely accurate; default 4.0",
        "<float>"
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
        params.symmetric = match matches.opt_get_default("symmetric", params.symmetric) {
            Ok(m) => m,
            Err(f) => panic!(f.to_string()),
        };
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
            Ok(m) => m,
            Err(f) => panic!(f.to_string()),
        };
        params.file_head = match matches.opt_get_default("overflow-file", params.file_head) {
            Ok(m) => m,
            Err(f) => panic!(f.to_string()),
        };
    }
    if params.max_product == 0 {
        params.max_product = estimate_max_prod(0.0, 1e-3);
    }
    params
}
