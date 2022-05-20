// Copyright 2022 Yashin Kirill
#ifndef MODULES_TASK_2_YASHIN_K_CRS_MULT_DOUBLE_OMP_SPARSE_MATRIX_MULTIPLICATION_OMP_H_
#define MODULES_TASK_2_YASHIN_K_CRS_MULT_DOUBLE_OMP_SPARSE_MATRIX_MULTIPLICATION_OMP_H_

#include <random>
#include <vector>

using Matrix = std::vector<std::vector<double>>;

class sparse_matrix {
    int rows;
    int columns;
    std::vector<double> values;
    std::vector<int> col_index;  // Column numbers for each item
    std::vector<int> row_index;  // Index of the beginning of each line

 public:
     sparse_matrix() : rows(0), columns(0) {}
     sparse_matrix(int _rows, int _cols, int _elemsCount) {
      rows = _rows;
      columns = _cols;
      values.resize(_elemsCount);
      col_index.resize(_elemsCount);
      row_index.resize(_rows + 1);
     }

     sparse_matrix(int _rows, int _cols, std::vector<double> _values,
        std::vector<int> _col_index, std::vector<int> _row_index) {
      rows = _rows;
      columns = _cols;
      values = _values;
      col_index = _col_index;
      row_index = _row_index;
    }

     explicit sparse_matrix(const Matrix& matrix) {
      rows = matrix.size();
      columns = matrix[0].size();

      int el_in_row = 0;
      row_index = {};
      col_index = {};
      values = {};
      row_index.reserve(rows * columns);
      col_index.reserve(rows * columns);
      values.reserve(rows + 1);
      row_index.push_back(0);

      for (int i = 0; i < rows; i++) {
          for (int j = 0; j < columns; j++) {
              if (std::fabs(matrix[i][j]) >= 0.00001) {
                  el_in_row++;
                  values.push_back(matrix[i][j]);
                  col_index.push_back(j);
              }
          }
          row_index.push_back(el_in_row);
      }
     }

     sparse_matrix(const sparse_matrix& matrix) {}

     ~sparse_matrix() {}

     bool operator== (const sparse_matrix& mat) const&;

     Matrix sparce_matrix_to_default();

     void printDefault();
     void printMatrix();

     friend sparse_matrix sparse_multiplication(const sparse_matrix& A,
      const sparse_matrix& B);
     friend sparse_matrix sparse_multiplication_omp(const sparse_matrix& A,
      const sparse_matrix& B);
};

sparse_matrix sparse_multiplication(const sparse_matrix& A,
  const sparse_matrix& B);
sparse_matrix sparse_multiplication_omp(const sparse_matrix& A,
  const sparse_matrix& B);
Matrix matrix_multiplication(const Matrix& A, const Matrix& B);
Matrix random_matrix(const int& rows, const int& columns, const int& freq);

#endif  // MODULES_TASK_2_YASHIN_K_CRS_MULT_DOUBLE_OMP_SPARSE_MATRIX_MULTIPLICATION_OMP_H_
