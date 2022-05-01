// Copyright 2022 Remizova Antonina
#include <gtest/gtest.h>
#include "../../../modules/task_2/remizova_a_hoar_batcher/hoar_batcher.h"

TEST(SEQ_hoar_batcher, can_create_rand_vec_10) {
    int size = 10;

    EXPECT_NO_THROW(getRandVector(size));
}

TEST(SEQ_hoar_batcher, can_create_rand_vec_10000) {
    int size = 10000;

    EXPECT_NO_THROW(getRandVector(size));
}

TEST(SEQ_hoar_batcher, can_not_create_rand_vec_with_negative_size) {
    int size = -10;

    EXPECT_ANY_THROW(getRandVector(size));
}

TEST(SEQ_hoar_batcher, can_hoar_sort_rand_vec) {
    int size = 231;
    std::vector<double> v = getRandVector(size);
    hoarSort(&v, 0, size - 1);
    EXPECT_EQ(1, isSortCorrectly(v));
}

TEST(SEQ_hoar_batcher, can_hoar_batcher_sort_zero_vec) {
    std::vector<double> v{ 0, 0, 1, 0, 0, 0, 0, 0 };
    int size = v.size();
    std::vector<double> res = hoarBatcherSeq(v, size);
    EXPECT_EQ(1, isSortCorrectly(res));
}

TEST(SEQ_hoar_batcher, can_hoar_batcher_sort_vec_with_even_size) {
    int size = 400;
    std::vector<double> v = getRandVector(size);
    std::vector<double> res = hoarBatcherSeq(v, v.size());
    EXPECT_EQ(1, isSortCorrectly(res));
}

TEST(SEQ_hoar_batcher, can_hoar_batcher_sort_vec_with_odd_size) {
    int size = 1331;
    std::vector<double> v = getRandVector(size);
    std::vector<double> res = hoarBatcherSeq(v, v.size());
    EXPECT_EQ(1, isSortCorrectly(res));
}

TEST(SEQ_hoar_batcher, can_hoar_batcher_sort_vec_with_double_elem) {
    int size = 9081;
    std::vector<double> v = getRandVector(size);
    v[740] = 4.5691;
    v[2943] = -17.89;
    std::vector<double> res = hoarBatcherSeq(v, v.size());
    EXPECT_EQ(1, isSortCorrectly(res));
}

//  parallel part

TEST(PAR_hoar_batcher, can_hoar_batcher_sort_zero_vec) {
    std::vector<double> v{ 0, 0, 1, 0, 0, 0, 0, 0 };
    int size = v.size();
    std::vector<double> res = hoarBatcherPar(v, size);
    EXPECT_EQ(1, isSortCorrectly(res));
}

TEST(PAR_hoar_batcher, can_hoar_batcher_sort_vec_with_even_size) {
    int size = 400;
    std::vector<double> v = getRandVector(size);
    std::vector<double> res = hoarBatcherPar(v, v.size());
    EXPECT_EQ(1, isSortCorrectly(res));
}

TEST(PAR_hoar_batcher, can_hoar_batcher_sort_vec_with_odd_size) {
    int size = 1331;
    std::vector<double> v = getRandVector(size);
    std::vector<double> res = hoarBatcherPar(v, v.size());
    EXPECT_EQ(1, isSortCorrectly(res));
}

TEST(PAR_hoar_batcher, can_hoar_batcher_sort_vec_with_double_elem) {
    int size = 9081;
    std::vector<double> v = getRandVector(size);
    v[740] = 4.5691;
    v[2943] = -17.89;
    std::vector<double> res = hoarBatcherPar(v, v.size());
    EXPECT_EQ(1, isSortCorrectly(res));
}

TEST(PAR_hoar_batcher, compare_seq_seq_sort_with_vec_size_10) {
    int size = 10;
    std::vector<double> v = getRandVector(size);

    double start_time, seq_time, par_time;
    start_time = omp_get_wtime();
    std::vector<double> res_seq = hoarBatcherSeq(v, v.size());
    seq_time = omp_get_wtime() - start_time;

    start_time = omp_get_wtime();
    std::vector<double> res_par = hoarBatcherSeq(v, v.size());
    par_time = omp_get_wtime() - start_time;

    std::cout << "res: " << seq_time / par_time;

    EXPECT_EQ(res_seq, res_par);
}

TEST(PAR_hoar_batcher, compare_seq_par_sort_with_vec_size_100) {
    int size = 10000;
    std::vector<double> v = getRandVector(size);
    
    double start_time, seq_time, par_time;
    start_time = omp_get_wtime();
    std::vector<double> res_seq = hoarBatcherSeq(v, v.size());
    seq_time = omp_get_wtime() - start_time;

    start_time = omp_get_wtime();
    std::vector<double> res_par = hoarBatcherPar(v, v.size());
    par_time = omp_get_wtime() - start_time;

    std::cout << "res: " << seq_time / par_time;

    EXPECT_EQ(res_seq, res_par);
}

TEST(PAR_hoar_batcher, compare_seq_par_sort_with_vec_size_1000) {
    int size = 100000;
    std::vector<double> v = getRandVector(size);

    double start_time, seq_time, par_time;
    start_time = omp_get_wtime();
    std::vector<double> res_seq = hoarBatcherSeq(v, v.size());
    seq_time = omp_get_wtime() - start_time;

    start_time = omp_get_wtime();
    std::vector<double> res_par = hoarBatcherPar(v, v.size());
    par_time = omp_get_wtime() - start_time;

    std::cout << "res: " << seq_time / par_time;

    EXPECT_EQ(res_seq, res_par);
}

TEST(PAR_hoar_batcher, compare_seq_par_sort_with_vec_size_1000000) {
    int size = 1000000;
    std::vector<double> v = getRandVector(size);

    double start_time, seq_time, par_time;
    start_time = omp_get_wtime();
    std::vector<double> res_seq = hoarBatcherSeq(v, v.size());
    seq_time = omp_get_wtime() - start_time;

    start_time = omp_get_wtime();
    std::vector<double> res_par = hoarBatcherPar(v, v.size());
    par_time = omp_get_wtime() - start_time;

    std::cout << "res: " << seq_time / par_time;

    EXPECT_EQ(res_seq, res_par);
}



int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
