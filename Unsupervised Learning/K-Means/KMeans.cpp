#include "KMeans.h"

// Constructor
template <typename T, std::size_t N>
KMeans<T, N>::KMeans(int k, const std::vector<Point<T, N>>& data) : k(k), data(data) {
    if (k <= 0) throw std::invalid_argument("Number of clusters must be greater than 0.");
    if (data.empty()) throw std::invalid_argument("Data set cannot be empty.");
    if (k > data.size()) throw std::invalid_argument("Number of clusters cannot exceed the number of data points.");
    initializeCentroids();
}

template <typename T, std::size_t N>
void KMeans<T, N>::run(int iterations) {
    for (int i = 0; i < iterations; ++i) {
        assignClusters();      // Assignation of data points to nearest centroids
        updateCentroids();
    }
}

// Return the centroids
template <typename T, std::size_t N>
const std::vector<Point<T, N>>& KMeans<T, N>::getCentroids() const {
    return centroids;
}

// Initialization of centroids by randomly selecting data points
template <typename T, std::size_t N>
void KMeans<T, N>::initializeCentroids() {
    centroids.clear();
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine engine(seed);

    std::vector<Point<T, N>> shuffledData = data;
    std::shuffle(shuffledData.begin(), shuffledData.end(), engine);
    centroids.assign(shuffledData.begin(), shuffledData.begin() + k);
}

// Euclidean distance between two points
template <typename T, std::size_t N>
T KMeans<T, N>::calculateDistance(const Point<T, N>& p1, const Point<T, N>& p2) const {
    T sum = 0;
    for (std::size_t i = 0; i < N; ++i) {
        T diff = p1.coordinates[i] - p2.coordinates[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Assignation of each data point to the nearest centroid
template <typename T, std::size_t N>
void KMeans<T, N>::assignClusters() {
    labels.resize(data.size());
    for (std::size_t i = 0; i < data.size(); ++i) {
        T minDist = std::numeric_limits<T>::max();
        int clusterIndex = 0;
        for (int j = 0; j < k; ++j) {
            T dist = calculateDistance(data[i], centroids[j]);
            if (dist < minDist) {
                minDist = dist;
                clusterIndex = j;
            }
        }
        labels[i] = clusterIndex;
    }
}

// Update centroids based on assigned clusters
template <typename T, std::size_t N>
void KMeans<T, N>::updateCentroids() {
    std::vector<int> count(k, 0);
    std::vector<Point<T, N>> newCentroids(k);

    for (int i = 0; i < k; ++i) {
        newCentroids[i] = Point<T, N>();
    }
    // Accumulate the sum of the coordinates of all points in each cluster
    for (std::size_t i = 0; i < data.size(); ++i) {
        // Add the coordinates of the current data point to the corresponding centroid
        for (std::size_t j = 0; j < N; ++j) {
            newCentroids[labels[i]].coordinates[j] += data[i].coordinates[j];
        }
        count[labels[i]]++;
    }

    for (int i = 0; i < k; ++i) {
        if (count[i] != 0) {
            for (std::size_t j = 0; j < N; ++j) {
                newCentroids[i].coordinates[j] /= count[i];
            }
        }
    }

    centroids = newCentroids;
}

// Explicit template instantiation to avoid linker issues
template class KMeans<double, 2>;
template class KMeans<float, 2>;
template class KMeans<double, 3>;
template class KMeans<float, 3>;

