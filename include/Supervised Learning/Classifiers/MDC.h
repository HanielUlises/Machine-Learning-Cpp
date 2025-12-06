#ifndef MINIMUM_DISTANCE_CLASSIFIER_H
#define MINIMUM_DISTANCE_CLASSIFIER_H

#include <iostream>
#include <array>
#include <vector>
#include <cmath>

// Represents a single data instance for classification, including its features and associated label.
struct instance {
    std::array<double, 2> features; // Feature vector containing two dimensions
    int label; // Class label associated with the instance
};

// Computes the Euclidean distance between two instances.
// The distance is calculated based on the features of both instances.
double calculate_distance(const instance& instance_1, const instance& instance_2) noexcept;

// Classifies a new instance by finding the closest training instance using the minimum distance classifier.
// This function returns the label of the nearest training instance.
int classify(const std::vector<instance>& training_data, const instance& new_instance) noexcept;

#endif // MINIMUM_DISTANCE_CLASSIFIER_H
