use super::sparse::SparseRecord;
use std::io;

pub struct Records<B> {
    pub filename: String,
    pub buffer: [u8; 20],
    pub reader: B,
}

impl<B: io::BufRead> Iterator for Records<B> {
    type Item = io::Result<SparseRecord>;

    fn next(&mut self) -> Option<io::Result<SparseRecord>> {
        match self.reader.read_exact(&mut self.buffer) {
            Ok(()) => {
                let record: bincode::Result<SparseRecord> = bincode::deserialize(&self.buffer);
                match record {
                    Ok(record) => Some(Ok(record)),
                    Err(_e) => None,
                }
            }
            Err(e) => Some(Err(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iterator() {
        let mut bytes = Vec::<u8>::with_capacity(100);
        for i in 1..5 {
            let rec = SparseRecord {
                row: i,
                col: i,
                val: i as f32,
            };
            if let Ok(b) = bincode::serialize(&rec) {
                bytes.extend(b);
            }
        }
        let cur = io::Cursor::new(bytes);
        let mut records = Records {
            filename: "memory".to_string(),
            buffer: [0; 20],
            reader: cur,
        };
        for i in 1..5 {
            let rec = SparseRecord {
                row: i,
                col: i,
                val: i as f32,
            };
            let n = records.next();
            assert!(n.is_some());
            if let Some(m) = n {
                assert!(m.is_ok());
                assert_eq!(m.unwrap(), rec);
            }
        }
    }
}
