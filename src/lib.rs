extern crate num;

use num::Float;
use std::cmp::Ordering;
use std::cmp::{max, min};

pub trait Serialize {
    // https://github.com/rust-lang/rust/issues/60551
    const BYTE_SIZE: usize;

    fn to_bytes(&self) -> Vec<u8>;

    fn from_bytes(bytes: Vec<u8>) -> Self;
}

impl Serialize for f32 {
    const BYTE_SIZE: usize = 4;
    fn to_bytes(&self) -> Vec<u8> {
        self.to_bits().to_le_bytes().to_vec()
    }

    fn from_bytes(byte_vec: Vec<u8>) -> Self {
        let mut bytes: [u8; 4] = Default::default();
        bytes.copy_from_slice(&byte_vec[..4]);
        f32::from_bits(u32::from_le_bytes(bytes))
    }
}

impl Serialize for f64 {
    const BYTE_SIZE: usize = 8;
    fn to_bytes(&self) -> Vec<u8> {
        self.to_bits().to_le_bytes().to_vec()
    }

    fn from_bytes(byte_vec: Vec<u8>) -> Self {
        let mut bytes: [u8; 8] = Default::default();
        bytes.copy_from_slice(&byte_vec[..8]);
        f64::from_bits(u64::from_le_bytes(bytes))
    }
}

#[derive(Debug, Copy, Clone)]
pub struct SparseRecord<F: Float + Serialize> {
    pub row: usize,
    pub col: usize,
    pub val: F,
}

impl<F: Float + Serialize> SparseRecord<F> {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut res = Vec::<u8>::new();
        res.extend(&self.row.to_le_bytes());
        res.extend(&self.col.to_le_bytes());
        res.extend(&self.val.to_bytes());
        res
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != 16 + F::BYTE_SIZE {
            None
        } else {
            let mut w1_bytes: [u8; 8] = [0; 8];
            w1_bytes.copy_from_slice(&bytes[0..8]);
            let mut w2_bytes: [u8; 8] = [0; 8];
            w2_bytes.copy_from_slice(&bytes[8..16]);

            Some(Self {
                row: usize::from_le_bytes(w1_bytes),
                col: usize::from_le_bytes(w2_bytes),
                val: F::from_bytes(bytes[16..16 + F::BYTE_SIZE].to_vec()),
            })
        }
    }
}

impl<F: Float + Serialize> PartialEq for SparseRecord<F> {
    fn eq(&self, other: &Self) -> bool {
        self.row == other.row && self.col == other.col
    }
}

impl<F: Float + Serialize> Eq for SparseRecord<F> {}

impl<F: Float + Serialize> PartialOrd for SparseRecord<F> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
}

impl<F: Float + Serialize> Ord for SparseRecord<F> {
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
/// ```
///     1  2  3  4  5
///     2  4  6  8 10
///     3  6  9 12 15
///     4  8 12 16 20
///     5 10 15 20 25
/// ```
/// If `max_prod = 10`, we will pack all of the elements where row x col <= 10 into
/// a single data vector, using the `row_offset` to quickly access any stored element.
/// ```
///  row_offset = [0         | 5          | 10     | 13   | 15   | 17]
///        data = [1 2 3 4 5 | 2 4 6 8 10 |  3 6 9 |  4 8 |  5 10]
/// ```
/// The remaining elements with products 12, 15, 16, 20 and 25 will be stored separately.
pub struct CornerMatrix<F: Float + Serialize> {
    pub max_size: usize,
    pub max_prod: usize,
    row_offset: Vec<usize>,
    data: Vec<F>,
}

impl<F: Float + Serialize> CornerMatrix<F> {
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
            data: vec![F::zero(); last],
        }
    }

    pub fn get(&mut self, row: usize, col: usize) -> &mut F {
        &mut self.data[self.row_offset[row] + col]
    }

    pub fn set(&mut self, row: usize, col: usize, value: F) {
        self.data[self.row_offset[row] + col] = value;
    }

    pub fn to_sparse(&self) -> Vec<SparseRecord<F>> {
        let mut result: Vec<SparseRecord<F>> = Vec::with_capacity(self.data.len());
        for row in 0..self.max_size {
            for col in 0..(self.row_offset[row + 1].saturating_sub(self.row_offset[row])) {
                let val = self.data[self.row_offset[row] + col];
                if val != F::zero() {
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

/// This function
pub fn estimate_max_prod<F: Float>(limit: F, tolerance: F) -> usize {
    let mut n: F = F::from(1e5).unwrap();
    let c: F = F::from(0.154_431_33).unwrap();
    while (limit - n * (n.ln() + c)).abs() > tolerance {
        n = limit / (n.ln() + c);
    }
    n.to_usize().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate() {
        let n = estimate_max_prod(1024.0, 1e-3);
        assert_eq!(n, 189);
    }

    #[test]
    fn test_matrix_creation() {
        let matrix = CornerMatrix::<f32>::new(5, 10);
        assert_eq!(matrix.row_offset, [0, 5, 10, 13, 15, 17]);
        assert_eq!(matrix.data.len(), 17);
    }

    #[test]
    fn test_matrix_getter_setter() {
        let mut matrix = CornerMatrix::<f64>::new(5, 10);
        assert!((*matrix.get(2, 3) - 0.0).abs() < 0.1);
        matrix.set(2, 3, 5.0);
        assert!((*matrix.get(2, 3) - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_to_sparse() {
        let mut matrix = CornerMatrix::<f32>::new(5, 10);
        *matrix.get(2, 1) += 1.0;

        let repr = matrix.to_sparse();
        println!("{:?}", matrix);
        println!("{:?}", repr);
        assert!(repr.len() == 1);
        assert!(
            repr == vec![
                SparseRecord::<f32> {
                    row: 2,
                    col: 1,
                    val: 1.0
                };
                1
            ]
        );
    }
}
