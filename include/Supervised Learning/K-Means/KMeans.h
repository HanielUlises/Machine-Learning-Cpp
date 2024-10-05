#ifndef KMEANS_H
#define KMEANS_H

#include <array>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <random>
#include <chrono>
#include <functional>
#include <stdexcept>

// Structure representing a point in N-dimensional space.
template <typename T, std::size_t N>
struct Point {
    std::array<T, N> coordinates;  // Coordinates of the point.

    Point() : coordinates{} {}

    // Initializes a point with the given coordinates.
    Point(const std::initializer_list<T>& values) {
        if (values.size() != N) {
            throw std::invalid_argument("Initializer list size does not match Point dimension.");
        }
        std::copy(values.begin(), values.end(), coordinates.begin());
    }
};

// Class implementing the KMeans clustering algorithm for N-dimensional points.
template <typename T, std::size_t N>
class KMeans {
public:
    // Constructs a KMeans instance with specified number of clusters and data points.
    KMeans(int num_clusters, const std::vector<Point<T, N>>& input_data);

    // Executes the KMeans algorithm for a specified number of iterations.
    void run(int iterations);

    // Returns the centroids of the clusters.
    const std::vector<Point<T, N>>& get_centroids() const;

private:
    int num_clusters_;                              // Number of clusters
    std::vector<Point<T, N>> centroids_;            // Centroids of clusters
    std::vector<Point<T, N>> data_;                 // Input data points
    std::vector<int> labels_;                        // Cluster labels for each data point

    // Initializes the centroids randomly from the input data.
    void initialize_centroids();

    // Computes the distance between two points.
    T calculate_distance(const Point<T, N>& point_a, const Point<T, N>& point_b) const;

    // Assigns each data point to the nearest cluster centroid.
    void assign_clusters();

    // Updates the cluster centroids based on current assignments.
    void update_centroids();
};

#endif // KMEANS_H
