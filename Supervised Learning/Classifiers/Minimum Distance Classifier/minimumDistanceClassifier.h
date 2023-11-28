#ifndef MINIMUM_DISTANCE_CLASSIFIER_H
#define MINIMUM_DISTANCE_CLASSIFIER_H

#include <iostream>
#include <array>
#include <vector>
#include <cmath>

// Data structure to represent the instances to be classified
struct Instance {
    std::array<double, 2> features;
    int label;
};

// Calculates the Euclidean distance between two given instances
double calculateDistance(const Instance& instance1, const Instance& instance2);

// Classifies a new instance using the minimum distance classifier
int classify(const std::vector<Instance>& trainingData, const Instance& newInstance);

#endif
