// Copyright 2022 Yashin Kirill
#include <gtest/gtest.h>
#include <tbb/tbb.h>
#include <iostream>
#include <vector>
#include "../../modules/task_3/yashin_k_crs_mult_double_tbb/sparse_matrix_multiplication_tbb.h"

TEST(Yashin_Kirill_Sparse_Matrix, Can_Multiply_Tbb_Random) {
    Matrix A(random_matrix(10, 10, 20));
    Matrix B(random_matrix(10, 10, 20));

    sparse_matrix sA(A);
    sparse_matrix sB(B);

    sparse_matrix result_sparse = sparse_multiplication_tbb(sA, sB);
    Matrix result = matrix_multiplication(A, B);

    sparse_matrix _result(result);

    ASSERT_TRUE(_result == result_sparse);
}

TEST(Yashin_Kirill_Sparse_Matrix, Test_Tbb_Matrix_Multiplication_Size_100) {
    Matrix A(random_matrix(100, 100, 20));
    Matrix B(random_matrix(100, 100, 20));

    sparse_matrix sA(A);
    sparse_matrix sB(B);

    Matrix result = matrix_multiplication(A, B);
    sparse_matrix sparse_result(result);

    tbb::tick_count t1_tbb = tbb::tick_count::now();
    sparse_matrix sparse_result_tbb = sparse_multiplication_tbb(sA, sB);
    tbb::tick_count t2_tbb = tbb::tick_count::now();

    tbb::tick_count t1_no_tbb = tbb::tick_count::now();
    sparse_matrix _sparse_result = sparse_multiplication(sA, sB);
    tbb::tick_count t2_no_tbb = tbb::tick_count::now();

    double acceleration = (t2_no_tbb - t1_no_tbb).seconds() / (t2_tbb - t1_tbb).seconds();
    std::cout << "TBB time for size 100: " << (t2_tbb - t1_tbb).seconds()
              << "\nNo TBB time for size 100: " << (t2_no_tbb - t1_no_tbb).seconds()
              << "\nAcceleration: " << acceleration << std::endl;

    ASSERT_TRUE(_sparse_result == sparse_result_tbb);
}

TEST(Yashin_Kirill_Sparse_Matrix, Test_Tbb_Matrix_Multiplication_Size_250) {
    Matrix A(random_matrix(100, 100, 20));
    Matrix B(random_matrix(100, 100, 20));

    sparse_matrix sA(A);
    sparse_matrix sB(B);

    Matrix result = matrix_multiplication(A, B);
    sparse_matrix sparse_result(result);

    tbb::tick_count t1_tbb = tbb::tick_count::now();
    sparse_matrix sparse_result_tbb = sparse_multiplication_tbb(sA, sB);
    tbb::tick_count t2_tbb = tbb::tick_count::now();

    tbb::tick_count t1_no_tbb = tbb::tick_count::now();
    sparse_matrix _sparse_result = sparse_multiplication(sA, sB);
    tbb::tick_count t2_no_tbb = tbb::tick_count::now();

    double acceleration = (t2_no_tbb - t1_no_tbb).seconds() / (t2_tbb - t1_tbb).seconds();
    std::cout << "TBB time for size 250: " << (t2_tbb - t1_tbb).seconds()
              << "\nNo TBB time for size 250: " << (t2_no_tbb - t1_no_tbb).seconds()
              << "\nAcceleration: " << acceleration << std::endl;

    ASSERT_TRUE(_sparse_result == sparse_result_tbb);
}

TEST(Yashin_Kirill_Sparse_Matrix, Test_Tbb_Matrix_Multiplication_Size_500) {
    Matrix A(random_matrix(500, 500, 20));
    Matrix B(random_matrix(500, 500, 20));

    sparse_matrix sA(A);
    sparse_matrix sB(B);

    Matrix result = matrix_multiplication(A, B);
    sparse_matrix sparse_result(result);

    tbb::tick_count t1_tbb = tbb::tick_count::now();
    sparse_matrix sparse_result_tbb = sparse_multiplication_tbb(sA, sB);
    tbb::tick_count t2_tbb = tbb::tick_count::now();

    tbb::tick_count t1_no_tbb = tbb::tick_count::now();
    sparse_matrix _sparse_result = sparse_multiplication(sA, sB);
    tbb::tick_count t2_no_tbb = tbb::tick_count::now();

    double acceleration = (t2_no_tbb - t1_no_tbb).seconds() / (t2_tbb - t1_tbb).seconds();
    std::cout << "TBB time for size 500: " << (t2_tbb - t1_tbb).seconds()
              << "\nNo TBB time for size 500: " << (t2_no_tbb - t1_no_tbb).seconds()
              << "\nAcceleration: " << acceleration << std::endl;

    ASSERT_TRUE(_sparse_result == sparse_result_tbb);
}

TEST(Yashin_Kirill_Sparse_Matrix, Test_Tbb_Matrix_Multiplication_Size_500_2) {
    Matrix A(random_matrix(500, 500, 10));
    Matrix B(random_matrix(500, 500, 10));

    sparse_matrix sA(A);
    sparse_matrix sB(B);

    Matrix result = matrix_multiplication(A, B);
    sparse_matrix sparse_result(result);

    tbb::tick_count t1_tbb = tbb::tick_count::now();
    sparse_matrix sparse_result_tbb = sparse_multiplication_tbb(sA, sB);
    tbb::tick_count t2_tbb = tbb::tick_count::now();

    tbb::tick_count t1_no_tbb = tbb::tick_count::now();
    sparse_matrix _sparse_result = sparse_multiplication(sA, sB);
    tbb::tick_count t2_no_tbb = tbb::tick_count::now();

    double acceleration = (t2_no_tbb - t1_no_tbb).seconds() / (t2_tbb - t1_tbb).seconds();
    std::cout << "TBB time for size 500: " << (t2_tbb - t1_tbb).seconds()
              << "\nNo TBB time for size 500: " << (t2_no_tbb - t1_no_tbb).seconds()
              << "\nAcceleration: " << acceleration << std::endl;

    ASSERT_TRUE(_sparse_result == sparse_result_tbb);
}
