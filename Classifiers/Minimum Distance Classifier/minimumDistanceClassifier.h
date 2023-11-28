// minimumDistanceClassifier.h

#include <iostream>
#include <array>
#include <vector>
#include <cmath>

// Data structure to represent the instances to be classified
template <size_t N>
struct Instance {
    std::array<double, N> features;
    int label;
};

// Calculates the Euclidean distance between two given instances
template <size_t N>
double calculateDistance(const Instance<N>& instance1, const Instance<N>& instance2);

// Classifies a new instance using the minimum distance classifier
template <size_t N>
int classify(const std::vector<Instance<N>>& trainingData, const Instance<N>& newInstance);