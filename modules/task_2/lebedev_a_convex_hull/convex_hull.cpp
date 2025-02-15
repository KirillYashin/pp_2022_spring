// Copyright 2022 Lebedev Alexey
#include "../../../modules/task_2/lebedev_a_convex_hull/convex_hull.h"
#include <omp.h>
#include <iostream>
#include <algorithm>
#include <numeric>

#define MIN_STEP 1000


bool cw(const cv::Point2d& p1, const cv::Point2d& p2, const cv::Point2d& p3) {
    return p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y) < 0;
}

bool ccw(const cv::Point2d& p1, const cv::Point2d& p2, const cv::Point2d& p3) {
    return p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y) > 0;
}

using points_iterator = std::vector<cv::Point2d>::iterator;
void convex_hull_impl(points_iterator begin, points_iterator end, std::vector<cv::Point2d>* output) {
    size_t size = end - begin;
    if (size <= 2) {
        output->resize(size);
        std::copy(begin, end, output->begin());
        return;
    }
    std::sort(begin, end, [](const cv::Point2d& p1, const cv::Point2d& p2) {
        return p1.x < p2.x || (p1.x == p2.x && p1.y < p2.y);
        });
    cv::Point2d p1 = *begin, p2 = *(end - 1);
    std::vector<cv::Point2d> up, down;
    up.push_back(p1);
    down.push_back(p1);
    for (size_t i = 1; i < size; ++i) {
        cv::Point2d p = *(begin + i);
        if (i == size - 1 || cw(p1, p, p2)) {
            while (up.size() >= 2 && !cw(up[up.size() - 2], up[up.size() - 1], p)) {
                up.pop_back();
            }
            up.push_back(p);
        }
        if (i == size - 1 || ccw(p1, p, p2)) {
            while (down.size() >= 2 && !ccw(down[down.size() - 2], down[down.size() - 1], p)) {
                down.pop_back();
            }
            down.push_back(p);
        }
    }
    output->clear();
    output->resize(up.size() + down.size() - 2);
    auto it = std::copy(up.begin(), up.end(), output->begin());
    std::copy(down.rbegin() + 1, down.rend() - 1, it);
}


size_t get_effective_num_threads(size_t size) {
    uint8_t add = size / MIN_STEP == 0 ? 1 : 0;
    return std::min(static_cast<int>(size / MIN_STEP + add), omp_get_max_threads());
}


void lab2::convex_hull(const cv::Mat& input, std::vector<cv::Point2d>* output, Version v) {
    std::vector<cv::Point2d> non_zeros;
    cv::findNonZero(input, non_zeros);
    lab2::convex_hull(&non_zeros, output, v);
}

void lab2::convex_hull(std::vector<cv::Point2d>* input, std::vector<cv::Point2d>* output, Version v) {
    size_t jobs = (v == Version::PARALLEL) ? get_effective_num_threads(input->size()) : 1;
    if (v != SEQUENTIAL) {
        size_t step = input->size() / jobs;
        std::vector<cv::Point2d> last(input->end() - input->size() % jobs, input->end());
        std::vector<std::vector<cv::Point2d>> hulls(jobs);
#pragma omp parallel num_threads(jobs) shared(input, step, hulls)
        {
            int tid = omp_get_thread_num();
            convex_hull_impl(input->begin() + tid * step, input->begin() + (tid + 1) * step, &hulls[tid]);
        }
        size_t size = std::accumulate(hulls.begin(), hulls.end(), 0,
            [](const size_t& _size, const std::vector<cv::Point2d>& v) {
                return _size + v.size();
            });
        std::vector<cv::Point2d> concat;
        concat.resize(size + last.size());
        auto it = std::copy(last.begin(), last.end(), concat.begin());
        for (const auto& hull : hulls) {
            it = std::copy(hull.begin(), hull.end(), it);
        }
        convex_hull_impl(concat.begin(), concat.end(), output);
    } else {
        convex_hull_impl(input->begin(), input->end(), output);
    }
}
