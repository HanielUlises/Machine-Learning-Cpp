#include "knn.h"

KNN::KNN(const std::vector<Point>& training_data) : training_data_(training_data) {}

//  Euclidean distance between two points.
double KNN::calculate_distance(const Point& point_a, const Point& point_b) const {
    return std::sqrt(std::pow(point_a.x - point_b.x, 2) + std::pow(point_a.y - point_b.y, 2));
}

// Classifies the input point based on the k-nearest neighbors.
Point KNN::classify(const Point& input_point, int k) const{
    std::vector<std::pair<double, int>> distances;

    // Calculate distances from the input point to all training data points.
    for (std::size_t i = 0; i < training_data_.size(); ++i) {
        double distance = calculate_distance(input_point, training_data_[i]);
        distances.emplace_back(distance, i);
    }

    // Sort distances in ascending order.
    std::sort(distances.begin(), distances.end());

    double sum_x = 0.0;
    double sum_y = 0.0;

    // Accumulate the coordinates of the k nearest neighbors.
    for (int i = 0; i < k; ++i) {
        sum_x += training_data_[distances[i].second].x;
        sum_y += training_data_[distances[i].second].y;
    }

    // Calculate the average position of the k nearest neighbors.
    Point result;
    result.x = sum_x / k;
    result.y = sum_y / k;

    return result;
}
