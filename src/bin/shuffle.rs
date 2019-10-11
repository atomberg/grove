#[macro_use]
extern crate log;
extern crate rand;

use env_logger;
use getopts::Options;
// use rand::distributions::Uniform;
use std::env;
use std::io::{self, BufReader, BufWriter, Write};

use grove::{record_io::Records, sparse::SparseRecord};

struct Params {
    memory_limit: usize,
    array_size: usize,
    temp_file: String,
}

fn main() {
    env_logger::init();
    let params = parse_args();
    let mut rng = rand::thread_rng();
    info!("Starting to shuffle with array_size={}", params.array_size);

    let mut shuffling_buffer = Vec::<SparseRecord<f32>>::with_capacity(params.array_size + 1);
    let mut reader = Records {
        buffer: [0; 20],
        filename: "stdin".to_string(),
        reader: BufReader::new(io::stdin()),
    };

    // Fill the buffer
    for _ in 0..params.array_size {
        if let Some(result) = reader.next() {
            if let Ok(record) = result {
                shuffling_buffer.push(record);
            }
        }
    }
    // shuffling_buffer.shuffle(&mut rng);
    // let between = Uniform::from(0..shuffling_buffer.len());

    let mut writer = BufWriter::new(io::stdout());
    for next in reader {
        let i = 0; // rng.gen_range(0, shuffling_buffer.len());
        if let Ok(record) = next {
            shuffling_buffer.push(record);
        }
        let record = shuffling_buffer.swap_remove(i);
        match writer.write_all(&record.to_bytes()) {
            Ok(n) => n,
            Err(e) => panic!("Could not write: {}", e.to_string()),
        };
    }
}

fn parse_args() -> Params {
    let args: Vec<String> = env::args().collect();
    let mut opts = Options::new();
    let mut params = Params {
        memory_limit: 0,
        array_size: 0,
        temp_file: "temp_shuffle".to_string(),
    };

    opts.optopt(
        "",
        "memory-limit",
        "Soft limit for memory consumption, in GB -- based on simple heuristic, so not extremely accurate; default 4.0",
        "<float>",
    );
    opts.optopt(
        "",
        "array-size",
        "Limit to length of the buffer which stores chunks of data to shuffle before writing to disk.",
        "<int>",
    );
    opts.optopt(
        "",
        "temp-file",
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
            opts.usage("Usage: ./shuffle [options] < cooccurrence.bin > cooccurrence.shuf.bin")
        );
        std::process::exit(0);
    } else {
        params.memory_limit = match matches.opt_get_default("memory-limit", params.memory_limit) {
            Ok(m) => m,
            Err(f) => panic!(f.to_string()),
        };
        params.array_size = match matches.opt_get_default("array-size", params.array_size) {
            Ok(m) => m,
            Err(f) => panic!(f.to_string()),
        };
        params.temp_file = match matches.opt_get_default("temp-file", params.temp_file) {
            Ok(m) => m,
            Err(f) => panic!(f.to_string()),
        };
    }
    params
}
