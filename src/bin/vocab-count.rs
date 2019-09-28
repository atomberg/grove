use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use twox_hash::RandomXxHashBuilder64;

fn main() {
    let mut args = env::args();
    let mut reader: Box<dyn BufRead> = match args.nth(1) {
        None => Box::new(BufReader::new(io::stdin())),
        Some(filename) => Box::new(BufReader::new(fs::File::open(filename).unwrap())),
    };
    let mut writer: Box<dyn Write> = match args.next() {
        None => Box::new(BufWriter::new(io::stdout())),
        Some(filename) => Box::new(BufWriter::new(fs::File::create(filename).unwrap())),
    };

    let map = hash_counts(&mut reader);
    for (word, count) in map.iter() {
        write!(writer, "{} {}\n", word, count).unwrap();
    }
}

fn hash_counts(reader: &mut Box<dyn BufRead>) -> HashMap<String, i32, RandomXxHashBuilder64> {
    let mut map: HashMap<String, i32, RandomXxHashBuilder64> = Default::default();
    let mut line = String::new();
    while reader.read_line(&mut line).unwrap() > 0 {
        process_line(&mut map, &line);
        line.clear();
    }
    return map;
}

fn process_line(map: &mut HashMap<String, i32, RandomXxHashBuilder64>, line: &str) {
    for word in line.trim_start().split_whitespace() {
        let counter = map.entry(word.to_string()).or_insert(0);
        *counter += 1;
    }
}
