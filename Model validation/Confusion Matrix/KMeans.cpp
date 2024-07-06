#include "KMeans.h"
#include <cmath>
#include <limits>
#include <numeric>
#include <algorithm>

KMeans::KMeans(int k, int max_iters) : k(k), max_iters(max_iters) {}

void KMeans::fit(const std::vector<std::vector<double>>& data) {
    // Initialize centroids to the first k points in the dataset
    centroids.assign(data.begin(), data.begin() + k);
    
    for (int iter = 0; iter < max_iters; ++iter) {
        // Assign points to the closest centroid
        std::vector<std::vector<std::vector<double>>> clusters(k);
        for (const auto& point : data) {
            int centroid_idx = closestCentroid(point);
            clusters[centroid_idx].push_back(point);
        }

        // Update centroids
        for (int i = 0; i < k; ++i) {
            if (!clusters[i].empty()) {
                centroids[i] = computeCentroid(clusters[i]);
            }
        }
    }
}

std::vector<int> KMeans::predict(const std::vector<std::vector<double>>& data) {
    std::vector<int> labels;
    for (const auto& point : data) {
        labels.push_back(closestCentroid(point));
    }
    return labels;
}

std::vector<std::vector<double>> KMeans::getCentroids() const {
    return centroids;
}

int KMeans::closestCentroid(const std::vector<double>& point) {
    int best_idx = 0;
    double best_dist = std::numeric_limits<double>::max();
    for (int i = 0; i < k; ++i) {
        double dist = 0.0;
        for (size_t j = 0; j < point.size(); ++j) {
            dist += std::pow(point[j] - centroids[i][j], 2);
        }
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = i;
        }
    }
    return best_idx;
}

std::vector<double> KMeans::computeCentroid(const std::vector<std::vector<double>>& points) {
    std::vector<double> centroid(points[0].size(), 0.0);
    for (const auto& point : points) {
        for (size_t i = 0; i < point.size(); ++i) {
            centroid[i] += point[i];
        }
    }
    for (double& val : centroid) {
        val /= points.size();
    }
    return centroid;
}
