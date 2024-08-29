#ifndef KMEANS_H
#define KMEANS_H

#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <random>
#include <chrono>
#include <functional>
#include <stdexcept>

// Generalized Point structure for N-dimensional space
template <typename T, std::size_t N>
struct Point {
    std::array<T, N> coordinates;

    Point() : coordinates{} {}

    Point(const std::initializer_list<T>& values) {
        if (values.size() != N) throw std::invalid_argument("Initializer list size does not match Point dimension.");
        std::copy(values.begin(), values.end(), coordinates.begin());
    }
};

// KMeans clustering algorithm for N-dimensional points
template <typename T, std::size_t N>
class KMeans {
public:
    KMeans(int k, const std::vector<Point<T, N>>& data);

    void run(int iterations);
    const std::vector<Point<T, N>>& getCentroids() const;

private:
    int k;                                  // Number of clusters
    std::vector<Point<T, N>> centroids;     // Centroids of clusters
    std::vector<Point<T, N>> data;          // Data points
    std::vector<int> labels;                // Cluster labels for each data point

    void initializeCentroids();
    T calculateDistance(const Point<T, N>& p1, const Point<T, N>& p2) const;
    void assignClusters();
    void updateCentroids();
};

#endif // KMEANS_H
