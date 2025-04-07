#include "KMeans.h"

// Constructor for the KMeans class.
// Initializes the number of clusters and data points, checking for validity of input parameters.
template <typename T, std::size_t N>
KMeans<T, N>::KMeans(int num_clusters, const std::vector<Point<T, N>>& input_data) 
    : num_clusters_(num_clusters), data_(input_data) {
    if (num_clusters <= 0) {
        throw std::invalid_argument("Number of clusters must be greater than 0.");
    }
    if (data_.empty()) {
        throw std::invalid_argument("Data set cannot be empty.");
    }
    if (num_clusters > data_.size()) {
        throw std::invalid_argument("Number of clusters cannot exceed the number of data points.");
    }
    initialize_centroids();
}

// Executes the KMeans algorithm for the specified number of iterations.
template <typename T, std::size_t N>
void KMeans<T, N>::run(int iterations) {
    for (int i = 0; i < iterations; ++i) {
        assign_clusters();  // Assigns data points to the nearest centroids.
        update_centroids(); // Updates the centroids based on current assignments.
    }
}

// Returns the centroids of the clusters.
template <typename T, std::size_t N>
const std::vector<Point<T, N>>& KMeans<T, N>::get_centroids() const {
    return centroids_;
}

// Initializes centroids by randomly selecting data points.
template <typename T, std::size_t N>
void KMeans<T, N>::initialize_centroids() {
    centroids_.clear();
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine engine(seed);

    std::vector<Point<T, N>> shuffled_data = data_;
    std::shuffle(shuffled_data.begin(), shuffled_data.end(), engine);
    centroids_.assign(shuffled_data.begin(), shuffled_data.begin() + num_clusters_);
}

// Computes the Euclidean distance between two points.
template <typename T, std::size_t N>
T KMeans<T, N>::calculate_distance(const Point<T, N>& point_a, const Point<T, N>& point_b) const {
    T sum = 0;
    for (std::size_t i = 0; i < N; ++i) {
        T diff = point_a.coordinates[i] - point_b.coordinates[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Assigns each data point to the nearest centroid.
template <typename T, std::size_t N>
void KMeans<T, N>::assign_clusters() {
    labels_.resize(data_.size());
    for (std::size_t i = 0; i < data_.size(); ++i) {
        T min_distance = std::numeric_limits<T>::max();
        int cluster_index = 0;
        for (int j = 0; j < num_clusters_; ++j) {
            T distance = calculate_distance(data_[i], centroids_[j]);
            if (distance < min_distance) {
                min_distance = distance;
                cluster_index = j;
            }
        }
        labels_[i] = cluster_index;
    }
}

// Updates centroids based on assigned clusters.
template <typename T, std::size_t N>
void KMeans<T, N>::update_centroids() {
    std::vector<int> count(num_clusters_, 0);
    std::vector<Point<T, N>> new_centroids(num_clusters_);

    // Accumulates the sum of the coordinates of all points in each cluster.
    for (std::size_t i = 0; i < data_.size(); ++i) {
        // Adds the coordinates of the current data point to the corresponding centroid.
        for (std::size_t j = 0; j < N; ++j) {
            new_centroids[labels_[i]].coordinates[j] += data_[i].coordinates[j];
        }
        count[labels_[i]]++;
    }

    // Averages the accumulated coordinates to update the centroids.
    for (int i = 0; i < num_clusters_; ++i) {
        if (count[i] != 0) {
            for (std::size_t j = 0; j < N; ++j) {
                new_centroids[i].coordinates[j] /= count[i];
            }
        }
    }

    centroids_ = new_centroids;
}

template class KMeans<double, 2>;
template class KMeans<float, 2>;
template class KMeans<double, 3>;
template class KMeans<float, 3>;
