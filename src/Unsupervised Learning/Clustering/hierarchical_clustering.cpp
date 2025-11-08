#include "HierarchicalClustering.h"
#include <limits>
#include <cmath>

// Initializes the clustering model with the specified number of clusters.
HierarchicalClustering::HierarchicalClustering(int num_clusters) : num_clusters_(num_clusters) {}

// Fits the hierarchical clustering model to the provided data matrix.
void HierarchicalClustering::fit(const Eigen::MatrixXd& data) {
    int n_samples = data.rows();
    labels_.resize(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        labels_[i] = i;  // Initialize each data point as its own cluster
    }

    Eigen::MatrixXd distances = Eigen::MatrixXd::Zero(n_samples, n_samples);
    for (int i = 0; i < n_samples; ++i) {
        for (int j = i + 1; j < n_samples; ++j) {
            distances(i, j) = distances(j, i) = compute_distance(data.row(i), data.row(j));
        }
    }

    while (n_samples > num_clusters_) {
        auto [cluster_a, cluster_b, min_dist] = find_closest_clusters(distances);
        
        // Update labels to reflect merging of clusters
        for (int i = 0; i < labels_.size(); ++i) {
            if (labels_[i] == cluster_b) {
                labels_[i] = cluster_a;  // Merge cluster b into cluster a
            }
        }
        
        update_distance_matrix(distances, cluster_a, cluster_b);
        --n_samples;  // Reduce the number of active clusters
    }
}

// Returns the cluster labels assigned to each data point.
std::vector<int> HierarchicalClustering::get_labels() const {
    return labels_;
}

// Computes the Euclidean distance between two points.
double HierarchicalClustering::compute_distance(const Eigen::VectorXd& point_a, const Eigen::VectorXd& point_b) const {
    return (point_a - point_b).norm();  // Returns the L2 norm (Euclidean distance)
}

// Finds the indices and distance of the two closest clusters.
std::tuple<int, int, double> HierarchicalClustering::find_closest_clusters(const Eigen::MatrixXd& distances) const {
    int cluster_a = -1;
    int cluster_b = -1;
    double min_dist = std::numeric_limits<double>::max();

    for (int i = 0; i < distances.rows(); ++i) {
        for (int j = i + 1; j < distances.cols(); ++j) {
            if (distances(i, j) < min_dist) {
                min_dist = distances(i, j);
                cluster_a = i;
                cluster_b = j;
            }
        }
    }

    return std::make_tuple(cluster_a, cluster_b, min_dist);
}

// Updates the distance matrix after merging two clusters.
void HierarchicalClustering::update_distance_matrix(Eigen::MatrixXd& distances, int cluster_a, int cluster_b) noexcept {
    for (int i = 0; i < distances.rows(); ++i) {
        if (i != cluster_a && i != cluster_b) {
            // Update the distance to the merged cluster
            distances(cluster_a, i) = distances(i, cluster_a) = std::min(distances(cluster_a, i), distances(cluster_b, i));
        }
    }
    // Mark the distances to the merged cluster as infinite
    distances.col(cluster_b).setConstant(std::numeric_limits<double>::max());
    distances.row(cluster_b).setConstant(std::numeric_limits<double>::max());
}
