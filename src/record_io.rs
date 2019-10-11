use std::io;

use super::sparse::SparseRecord;

pub struct Records<B> {
    pub filename: String,
    pub buffer: [u8; 20],
    pub reader: B,
}

impl<B: io::BufRead> Iterator for Records<B> {
    type Item = io::Result<SparseRecord<f32>>;

    fn next(&mut self) -> Option<io::Result<SparseRecord<f32>>> {
        match self.reader.read_exact(&mut self.buffer) {
            Ok(()) => match SparseRecord::from_bytes(&self.buffer) {
                Some(record) => Some(Ok(record)),
                None => None,
            },
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
            bytes.extend(rec.to_bytes());
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
            if let Some(record) = records.next() {
                if let Ok(record) = record {
                    assert_eq!(record, rec);
                }
            }
        }
    }
}
