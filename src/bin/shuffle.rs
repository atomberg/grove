#[macro_use]
extern crate log;

use env_logger;
use getopts::Options;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::env;
use std::io::{self, BufReader, BufWriter, Write};

use grove::{record_io::Records, sparse::SparseRecord};

struct Params {
    array_size: usize,
}

fn main() {
    env_logger::init();
    info!("Starting");
    let params = parse_args();
    info!("Starting to shuffle with array_size={}", params.array_size);

    let mut shuffling_buffer = Vec::<SparseRecord>::with_capacity(params.array_size + 1);
    let mut reader = Records {
        buffer: [0; 20],
        filename: "stdin".to_string(),
        reader: BufReader::new(io::stdin()),
    };

    // Fill the buffer
    let mut counter = 0;
    for _ in 0..params.array_size {
        if let Some(result) = reader.next() {
            if let Ok(record) = result {
                counter += 1;
                shuffling_buffer.push(record);
            }
        }
    }
    info!("Filled the buffer with {} records, shuffling now", counter);
    shuffling_buffer.shuffle(&mut thread_rng());

    info!("Buffer shuffled, let's go!");
    let mut writer = BufWriter::new(io::stdout());
    counter = 0;
    for next in reader {
        let i = 0; // rng.gen_range(0, shuffling_buffer.len());
        if let Ok(record) = next {
            shuffling_buffer.push(record);
        }
        if shuffling_buffer.is_empty() {
            break;
        }
        let record = shuffling_buffer.swap_remove(i);
        let bytes = match bincode::serialize(&record) {
            Ok(b) => b,
            Err(e) => panic!("Could not serialize record: {}", e.to_string()),
        };
        match writer.write_all(&bytes) {
            Ok(n) => {
                counter += 1;
                n
            }
            Err(e) => panic!("Could not write: {}", e.to_string()),
        };
    }
    info!("Wrote out {} records.", counter);
}

fn parse_args() -> Params {
    let args: Vec<String> = env::args().collect();
    let mut opts = Options::new();
    let mut params = Params { array_size: 0 };

    opts.optopt(
        "",
        "array-size",
        "Limit to length of the buffer which stores chunks of data to shuffle before writing to disk.",
        "<int>",
    );
    opts.optflag("h", "help", "print this help menu");
    let matches = match opts.parse(&args[1..]) {
        Ok(m) => m,
        Err(f) => panic!(f.to_string()),
    };
    if matches.opt_present("help") {
        print!(
            "{}",
            opts.usage("Usage: ./shuffle [options] < cooccurrence.bin > cooccurrence.shuf.bin")
        );
        std::process::exit(0);
    } else {
        params.array_size = match matches.opt_get_default("array-size", params.array_size) {
            Ok(m) => m,
            Err(f) => panic!(f.to_string()),
        };
    }
    params
}
