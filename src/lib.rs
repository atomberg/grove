use std::cmp::{max, min};

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
///  row_offset = [0         | 5          | 10     | 13   | 15   ]
///        data = [1 2 3 4 5 | 2 4 6 8 10 |  3 6 9 |  4 8 |  5 10]
/// ```
/// The remaining elements with products 12, 15, 16, 20 and 25 will be stored separately.
struct CornerMatrix<T: Copy> {
    max_size: usize,
    max_prod: usize,
    row_offset: Vec<usize>,
    data: Vec<T>,
}

impl<T: Copy> CornerMatrix<T> {
    pub fn new(max_size: usize, max_prod: usize, initial_value: T) -> Self {
        let mut last: usize = 0;
        let mut row_offset = Vec::<usize>::with_capacity(max_size);
        row_offset.push(last);

        for i in 1..max_size {
            last += min(max_prod / i, max_size);
            row_offset.push(last);
        }
        CornerMatrix {
            max_size,
            max_prod,
            row_offset,
            data: vec![initial_value; last + min(max_prod / max_size, max_size)],
        }
    }

    pub fn get(&self, row: usize, col: usize) -> T {
        self.data[self.row_offset[row] + col]
    }

    pub fn set(&mut self, row: usize, col: usize, value: T) {
        self.data[self.row_offset[row] + col] = value;
    }
}

#[derive(Debug)]
/// A symmetic variant of a partially sparse matrix.
struct SymmetricCornerMatrix<T: Copy> {
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
fn estimate_max_prod(limit: f32, tolerance: f32) -> usize {
    let mut n: f32 = 1e5;
    let c: f32 = 0.154_431_33;
    while (limit - n * (n.ln() + c)).abs() > tolerance {
        n = limit / (n.ln() + c);
    }
    n as usize
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
        let matrix = CornerMatrix::<f32>::new(5, 10, 0.0);
        assert_eq!(matrix.row_offset, [0, 5, 10, 13, 15]);
        assert_eq!(matrix.data.len(), 17);
    }

    #[test]
    fn test_matrix_getter_setter() {
        let mut matrix = CornerMatrix::<f64>::new(5, 10, 0.0);
        assert_eq!(matrix.get(2, 3), 0.0);
        matrix.set(2, 3, 5.0);
        assert_eq!(matrix.get(2, 3), 5.0);
    }
}
