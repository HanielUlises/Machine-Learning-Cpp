#ifndef KNN_H
#define KNN_H

#include <vector>
#include <cmath>
#include <algorithm>

// Structure representing a point in 2D space.
struct Point {
    double x; // X-coordinate of the point.
    double y; // Y-coordinate of the point.
};

// Class implementing the K-Nearest Neighbors algorithm.
class KNN {
public:
    // Constructs the KNN classifier with the provided training data.
    explicit KNN(const std::vector<Point>& training_data);

    // Classifies the given point based on the k-nearest neighbors.
    Point classify(const Point& input_point, int k) const;

private:
    // Calculates the Euclidean distance between two points.
    double calculate_distance(const Point& point_a, const Point& point_b) const;

    // Collection of training data points.
    std::vector<Point> training_data_;
};

#endif // KNN_H
