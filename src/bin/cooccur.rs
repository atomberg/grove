extern crate getopts;

use getopts::Options;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use twox_hash::RandomXxHashBuilder64;

use ::CornerMatrix;


struct CooccurRecord (u32, u32, f32);

struct CooccurRecordWithId (u32, u32, f32, u32);

struct Params {
    window_size: usize,
    symmetric: bool,
    memory_limit: f32,
    max_product: i32,
    vocab_file: String,
    file_head: String,
    overflow_length: usize,
    verbose: usize
}


fn main() {
    let params = parse_args();
    let vocab_hash = read_vocab(params.vocab_file);

    
    let mut reader = BufReader::new(io::stdin());
    let mut writer = BufWriter::new(io::stdout());

    let n = 189;
    
    let table = CornerMatrix::<f32>::new(5, 10, 0.0);
    
}

fn read_vocab(vocab_filename: String) -> HashMap<String, i32, RandomXxHashBuilder64> {

    let vocab_file = match fs::File::open(&vocab_filename) {
        Ok(file) => file,
        Err(e) => panic!("Couldn't open {}: {}", vocab_filename, e.to_string())
    };
    let mut reader = BufReader::new(vocab_file);

    let mut map: HashMap<String, i32, RandomXxHashBuilder64> = Default::default();
    let mut line = String::new();
    let mut j = 1;
    while reader.read_line(&mut line).unwrap() > 0 {
        let words: Vec<&str> = line.split_whitespace().collect();
        map.insert(words[0].to_string(), j);
        line.clear();
        j += 1;
    }
    map
}

fn process_line() {

}

fn compute_cooc(words: Vec<u32>, window_size: usize, ) {
    for j in 0..(words.len() - 1) {
        let cooc: f32 = 0.0;
        let lower_bound = j - window_size;
        if lower_bound < 0 {
            lower_bound = 0;
        }
        for k in (j - 1)..lower_bound {
            cooc += 1.0 / (j - k) as f32;
        }
    }
    for window in words.windows(window_size) {
        let trg_word: u32 = match window.last() {
            None => 0:u32,
            Some(&x) => x
        };
        for ctx_word in window
        
    }
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
        verbose: 2
    };

    let args: Vec<String> = env::args().collect();

    let mut opts = Options::new();
    opts.optopt(
        "v",
        "verbose",
        "verbosity level (default 2)",
        "<int>"
    );
    opts.optopt(
        "symm",
        "symmetric",
        "if false, only use left context; if true, use left and right (default true)",
        "<bool>"
    );
    opts.optopt(
        "window",
        "window-size",
        "Number of context words to the left (and to the right, if symmetric=1); default 15",
        "<int>"
    );
    opts.optopt(
        "vocab",
        "vocab-file",
        "File containing vocabulary (truncated unigram counts, produced by 'vocab_count'); default vocab.txt",
        "<file>"
    );
    opts.optopt(
        "mem",
        "memory-limit",
        "Soft limit for memory consumption, in GB -- based on simple heuristic, so not extremely accurate; default 4.0",
        "<float>"
    );
    opts.optopt(
        "prod",
        "max-product",
        "Limit the size of dense cooccurrence array by specifying the max product <int> of 
        the frequency counts of the two cooccurring words. This value overrides that 
        which is automatically produced by '-memory'. 
        Typically only needs adjustment for use with very large corpora.", 
        "<int>"
    );
    opts.optopt(
        "tmp",
        "overflow-file",
        "Filename, excluding extension, for temporary file; default overflow",
        "<file>"
    );
    opts.optflag(
        "h",
        "help",
        "print this help menu"
    );
    let matches = match opts.parse(&args[1..]) {
        Ok(m) => { m }
        Err(f) => { panic!(f.to_string()) }
    };
    if matches.opt_present("help") {
        let brief = format!("Usage: ./cooccur [options] < corpus.txt > cooccurrences.bin");
        print!("{}", opts.usage(&brief));
    } else {
        params.verbose = match matches.opt_get_default("verbose", params.verbose) {
            Ok(m) => { m }
            Err(f) => { panic!(f.to_string()) }
        };
        params.symmetric = match matches.opt_get_default("symmetric", params.symmetric) {
            Ok(m) => { m }
            Err(f) => { panic!(f.to_string()) }
        };
        params.window_size = match matches.opt_get_default("window-size", params.window_size) {
            Ok(m) => { m }
            Err(f) => { panic!(f.to_string()) }
        };
        params.vocab_file = match matches.opt_get_default("vocab-file", params.vocab_file) {
            Ok(m) => { m }
            Err(f) => { panic!(f.to_string()) }
        };
        params.memory_limit = match matches.opt_get_default("memory-limit", params.memory_limit) {
            Ok(m) => { m }
            Err(f) => { panic!(f.to_string()) }
        };
        params.max_product = match matches.opt_get_default("max-product", params.max_product) {
            Ok(m) => { m }
            Err(f) => { panic!(f.to_string()) }
        };
        params.file_head = match matches.opt_get_default("overflow-file", params.file_head) {
            Ok(m) => { m }
            Err(f) => { panic!(f.to_string()) }
        };
    }
    params   
}