extern crate getopts;

#[macro_use]
extern crate log;

use env_logger;
use getopts::Options;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::env;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use twox_hash::RandomXxHashBuilder64;

struct Params {
    min_count: i32,
    max_vocab: usize,
    verbose: usize,
}

fn main() {
    env_logger::init();
    let params = parse_args();

    info!("Building vocabulary");
    let mut reader = BufReader::new(io::stdin());
    let map = hash_counts(&mut reader);
    let mut vocab = Vec::<(String, i32)>::with_capacity(map.len());
    for (word, count) in map {
        vocab.push((word, count));
    }

    if params.max_vocab > 0 && params.max_vocab < vocab.len() {
        vocab.sort_unstable_by_key(|(_word, count)| 1 - count);
        vocab.truncate(params.max_vocab);
    }
    vocab.sort_by(
        |(word_1, count_1), (word_2, count_2)| match count_1.cmp(&count_2) {
            Ordering::Less => Ordering::Greater,
            Ordering::Equal => word_1.cmp(&word_2),
            Ordering::Greater => Ordering::Less,
        },
    );

    let mut writer = BufWriter::new(io::stdout());
    let mut counter: usize = 0;
    for (word, count) in vocab {
        if count < params.min_count {
            break;
        }
        counter += 1;
        writeln!(writer, "{} {}", word, count).unwrap();
    }
    info!("Using vocabulary of size {}", counter);
}

fn hash_counts(reader: &mut dyn BufRead) -> HashMap<String, i32, RandomXxHashBuilder64> {
    let mut map: HashMap<String, i32, RandomXxHashBuilder64> = Default::default();
    let mut line = String::new();
    while reader.read_line(&mut line).unwrap() > 0 {
        for word in line.trim_start().split_whitespace() {
            let counter = map.entry(word.to_string()).or_insert(0);
            *counter += 1;
        }
        line.clear();
    }
    map
}

fn parse_args() -> Params {
    let args: Vec<String> = env::args().collect();
    let mut opts = Options::new();
    let mut params = Params {
        min_count: 0,
        max_vocab: 0,
        verbose: 2,
    };
    opts.optopt("", "verbose", "verbosity level (default 2)", "INT");
    opts.optopt(
        "", "max-vocab",
        "Upper bound on vocabulary size, i.e. keep the <max-vocab> most frequent words. \
        The minimum frequency words are randomly sampled so as to obtain an even distribution over the alphabet.",
        "INT"
    );
    opts.optopt(
        "",
        "min-count",
        "Lower limit such that words which occur fewer than <min-count> times are discarded.",
        "INT",
    );
    opts.optflag("h", "help", "print this help menu");
    let matches = match opts.parse(&args[1..]) {
        Ok(m) => m,
        Err(f) => panic!(f.to_string()),
    };
    if matches.opt_present("help") {
        print!(
            "{}",
            opts.usage("Usage: ./vocab_count [options] < corpus.txt > vocab.txt")
        );
        std::process::exit(0);
    } else {
        params.verbose = match matches.opt_get_default("verbose", params.verbose) {
            Ok(m) => m,
            Err(f) => panic!(f.to_string()),
        };
        params.max_vocab = match matches.opt_get_default("max-vocab", params.max_vocab) {
            Ok(m) => m,
            Err(f) => panic!(f.to_string()),
        };
        params.min_count = match matches.opt_get_default("min-count", params.min_count) {
            Ok(m) => m,
            Err(f) => panic!(f.to_string()),
        };
    }
    params
}
