use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::cmp::{max, min};

/// A serializable (row, col, val) triple for representing sparse matrix elements.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct SparseRecord {
    pub row: usize,
    pub col: usize,
    pub val: f32,
}

impl PartialEq for SparseRecord {
    fn eq(&self, other: &Self) -> bool {
        self.row == other.row && self.col == other.col
    }
}

impl Eq for SparseRecord {}

impl PartialOrd for SparseRecord {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
}

impl Ord for SparseRecord {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.row.cmp(&other.row) {
            Ordering::Equal => self.col.cmp(&other.col),
            x => x,
        }
    }
}

#[derive(Debug)]
/// A partially sparse matrix.
///
/// This represents the dense upper left corner of an otherwise sparse matrix, such as the bigram table in GLoVe.
/// We define the boundaries of the dense part by the maximum product of row * column indices, starting from 1.
///
/// For example for `max_size = 5`, the product of indices is as follows:
/// ```ignore
///     1  2  3  4  5
///     2  4  6  8 10
///     3  6  9 12 15
///     4  8 12 16 20
///     5 10 15 20 25
/// ```
/// If `max_prod = 10`, we will pack all of the elements where row x col <= 10 into
/// a single data vector, using the `row_offset` to quickly access any stored element.
/// ```ignore
///  row_offset = [0         | 5          | 10     | 13   | 15   | 17]
///        data = [1 2 3 4 5 | 2 4 6 8 10 |  3 6 9 |  4 8 |  5 10]
/// ```
/// The remaining elements with products 12, 15, 16, 20 and 25 will be stored separately.
pub struct CornerMatrix {
    pub max_size: usize,
    pub max_prod: usize,
    row_offset: Vec<usize>,
    data: Vec<f32>,
}

impl CornerMatrix {
    pub fn new(max_size: usize, max_prod: usize) -> Self {
        let mut last: usize = 0;
        let mut row_offset = Vec::<usize>::with_capacity(max_size);
        row_offset.push(last);

        for i in 1..max_size {
            last += min(max_prod / i, max_size);
            row_offset.push(last);
        }
        last += min(max_prod / max_size, max_size);
        row_offset.push(last);
        CornerMatrix {
            max_size,
            max_prod,
            row_offset,
            data: vec![0f32; last],
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> usize {
        let mut counter: usize = 0;
        for e in self.data.iter() {
            if *e > 0f32 {
                counter += 1;
            }
        }
        counter
    }

    pub fn get(&mut self, row: usize, col: usize) -> &mut f32 {
        &mut self.data[self.row_offset[row] + col]
    }

    pub fn to_sparse(&self) -> Vec<SparseRecord> {
        let mut result: Vec<SparseRecord> = Vec::with_capacity(self.data.len());
        for row in 0..self.max_size {
            for col in 0..(self.row_offset[row + 1].saturating_sub(self.row_offset[row])) {
                let val = self.data[self.row_offset[row] + col];
                if val != 0f32 {
                    result.push(SparseRecord { row, col, val });
                }
            }
        }
        result
    }
}

#[derive(Debug)]
/// A symmetic variant of a partially sparse matrix.
pub struct SymmetricCornerMatrix<T: Copy> {
    max_size: usize,
    max_prod: usize,
    row_offset: Vec<usize>,
    data: Vec<T>,
}

impl<T: Copy> SymmetricCornerMatrix<T> {
    pub fn new(max_size: usize, max_prod: usize, initial_value: T) -> Self {
        let mut last: usize = 0;
        let mut row_offset = Vec::<usize>::with_capacity(max_size);
        row_offset.push(last);
        for i in 1..max_size {
            last += max(min(max_prod / i, max_size), 0);
            row_offset.push(last);
        }
        SymmetricCornerMatrix {
            max_size,
            max_prod,
            row_offset,
            data: vec![initial_value; last],
        }
    }

    pub fn get(&self, row: usize, col: usize) -> T {
        if row <= col {
            self.data[self.row_offset[row] + col]
        } else {
            self.data[self.row_offset[col] + row]
        }
    }

    pub fn set(&mut self, row: usize, col: usize, value: T) {
        if row <= col {
            self.data[self.row_offset[row] + col] = value;
        } else {
            self.data[self.row_offset[col] + row] = value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_from_bytes_f32() {
        let a = SparseRecord {
            row: 1,
            col: 2,
            val: 3.0 as f32,
        };
        let mut buf: [u8; 12] = [0; 12];
        if let Ok(b) = bincode::serialize(&a) {
            buf.copy_from_slice(&b);
            let c: SparseRecord = bincode::deserialize(&buf).unwrap();
            assert_eq!(a, c);
        } else {
            unreachable!();
        }
    }

    #[test]
    fn test_matrix_creation() {
        let matrix = CornerMatrix::new(5, 10);
        assert_eq!(matrix.row_offset, [0, 5, 10, 13, 15, 17]);
        assert_eq!(matrix.data.len(), 17);
        assert!(matrix.is_empty());
    }

    #[test]
    fn test_matrix_getter_setter() {
        let mut matrix = CornerMatrix::new(5, 10);
        assert!((*matrix.get(2, 3) - 0.0).abs() < 0.1);
        *matrix.get(2, 3) += 5.0;
        assert!((*matrix.get(2, 3) - 5.0).abs() < 0.1);
        assert_eq!(matrix.len(), 1);
    }

    #[test]
    fn test_to_sparse() {
        let mut matrix = CornerMatrix::new(5, 10);
        *matrix.get(2, 1) += 1.0;

        let repr = matrix.to_sparse();
        assert!(repr.len() == matrix.len());
        assert!(
            repr == vec![
                SparseRecord {
                    row: 2,
                    col: 1,
                    val: 1.0
                };
                1
            ]
        );
    }
}
