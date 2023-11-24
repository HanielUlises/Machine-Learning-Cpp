#include <iostream>
#include <vector>
#include <cmath>

template <size_t N>
struct Instance {
    std::array<double, N> features;
    int label;
};

template <size_t N>
double calculateDistance(const Instance<N>& instance1, const Instance<N>& instance2);

template <size_t N>
int classify(const std::vector<Instance<N>>& trainingData, const Instance<N>& newInstance);