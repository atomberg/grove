use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::{BufRead, BufReader};

fn main() {
    let args: Vec<String> = env::args().collect();
    let filename = &args[1];

    let file = fs::File::open(filename).unwrap();
    let mut reader = BufReader::new(file);

    let mut map = HashMap::new();
    let mut buf = String::new();
    while reader.read_line(&mut buf).unwrap() > 0 {
        for word in buf.to_string().clone().split_whitespace() {
            let key = word.to_lowercase();
            let counter = map.entry(key).or_insert(0);
            *counter += 1;
        }
        buf.clear();
    }

    for (word, count) in map.iter() {
        println!("{}: {}", word, count);
    }
}

// fn count(text: String) {}
