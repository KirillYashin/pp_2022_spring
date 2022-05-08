// Copyright 2022 Yashin Kirill

#include <tbb/tbb.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include "../../modules/task_3/yashin_k_crs_mult_double_tbb/sparse_matrix_multiplication_tbb.h"

bool sparse_matrix::operator== (const sparse_matrix& matrix) const& {
    if (rows != matrix.rows || columns != matrix.columns)
        return false;

    for (int i = 0; i < rows; i++) {
        if (row_index[i] != matrix.row_index[i])
            return false;
    }

    for (int i = 0; i < columns; i++) {
        if (col_index[i] != matrix.col_index[i])
            return false;
    }

    for (size_t i = 0; i < values.size(); i++) {
       if (std::fabs(matrix.values[i] - values[i]) > 0.01)
            return false;
    }
    return true;
}

Matrix sparse_matrix::sparce_matrix_to_default() {
    Matrix result(rows, std::vector<double>(columns, 0.0));

    int temp_column = 0;
    for (int i = 0; i < rows; i++) {
        int temp_row = row_index[i + 1] - row_index[i];
        while (temp_row) {
            result[i][col_index[temp_column]] = values[temp_column];
            temp_row -= 1;
            temp_column += 1;
        }
    }
    return result;
}

Matrix matrix_multiplication(const Matrix& A, const Matrix& B) {
    Matrix result(A.size());
    for (size_t i = 0; i < result.size(); i++)
        result[i].resize(B[0].size());

    for (size_t i = 0; i < A.size(); i++) {
        for (size_t j = 0; j < B[0].size(); j++) {
            result[i][j] = 0;
            for (size_t k = 0; k < A[0].size(); k++)
                result[i][j] += A[i][k] * B[k][j];
        }
    }
    return result;
}

sparse_matrix sparse_multiplication(const sparse_matrix& A, const sparse_matrix& B) {
    sparse_matrix result;
    result.rows = A.rows;
    result.columns = B.columns;
    result.row_index.push_back(0);
    std::vector<double> temp_result_row(B.columns, 0);

    for (int i = 0; i < A.rows; i++) {
        for (int j = A.row_index[i]; j < A.row_index[i + 1]; j++) {
            int temp_column {A.col_index[j]};
            for (int k = B.row_index[temp_column]; k < B.row_index[temp_column + 1]; k++)
                temp_result_row[B.col_index[k]] += A.values[j] * B.values[k];
        }
        for (int k = 0; k < B.columns; k++) {
            if (temp_result_row[k]) {
                result.values.push_back(temp_result_row[k]);
                result.col_index.push_back(k);
                temp_result_row[k] = 0;
            }
        }
        result.row_index.push_back(result.values.size());
    }
    return result;
}

sparse_matrix sparse_multiplication_tbb(const sparse_matrix& A, const sparse_matrix& B) {
    int grain_size = 10;

    sparse_matrix result;
    result.rows = A.rows;
    result.columns = B.columns;

    result.row_index.resize(result.rows + 1);
    std::vector<int> temp_result_row(A.rows + 1, 0);
    std::vector<int>*    temp_result_columns = new std::vector<int>[result.rows];
    std::vector<double>* temp_result_values = new std::vector<double>[result.rows];

    tbb::parallel_for(tbb::blocked_range<int>(0, result.rows, grain_size), [&](tbb::blocked_range<int> r) {
        int temp_column = 0;
        for (int i = r.begin(); i < r.end(); i++) {
            std::vector<double> temp_result(A.rows + 1, 0);
            for (int j = A.row_index[i]; j < A.row_index[i + 1]; j++) {
                temp_column = A.col_index[j];
                for (int k = B.row_index[temp_column]; k < B.row_index[temp_column + 1]; k++)
                    temp_result[B.col_index[k]] += A.values[j] * B.values[k];
            }
            for (int k = 0; k < A.rows; k++) {
                if (temp_result[k]) {
                    temp_result_values[i].push_back(temp_result[k]);
                    temp_result_columns[i].push_back(k);
                    temp_result_row[i]++;
                }
            }
        }
    });

    int count = 0;
    int temp_rows = 0;

    for (int i = 0; i < result.rows; i++) {
        int tmp = temp_result_row[i];
        result.row_index[i] = temp_rows;
        temp_rows += tmp;
    }

    result.row_index[A.rows] = temp_rows;
    result.col_index.resize(temp_rows);
    result.values.resize(temp_rows);
    for (int i = 0; i < result.rows; i++) {
        size_t size = temp_result_columns[i].size();
        if (size) {
            memcpy(&result.col_index[count], &temp_result_columns[i][0], size * sizeof(int));
            memcpy(&result.values[count], &temp_result_values[i][0], size * sizeof(double));
            count += size;
        }
    }

    delete[]temp_result_columns;
    delete[]temp_result_values;

    return result;
}

Matrix random_matrix(const int& rows, const int& columns, const int& freq) {
    std::random_device rand{};
    std::mt19937 mt {rand()};
    std::uniform_real_distribution<double> rand_values {0.0, 10.0};
    std::uniform_int_distribution<int> rand_probability {0, 100};
    Matrix result{};
    result.resize(rows);
    for (int i = 0; i < rows; i++)
        result[i].resize(columns);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            if (rand_probability(mt) <= freq)
                result[i][j] = rand_values(mt);
        }
    }
    return result;
}
