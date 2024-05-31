#include "HierarchicalClustering.h"
#include <limits>
#include <cmath>

HierarchicalClustering::HierarchicalClustering(int num_clusters) : num_clusters(num_clusters) {}

void HierarchicalClustering::fit(const Eigen::MatrixXd& data) {
    int n_samples = data.rows();
    labels.resize(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        labels[i] = i;
    }

    Eigen::MatrixXd distances = Eigen::MatrixXd::Zero(n_samples, n_samples);
    for (int i = 0; i < n_samples; ++i) {
        for (int j = i + 1; j < n_samples; ++j) {
            distances(i, j) = distances(j, i) = compute_distance(data.row(i), data.row(j));
        }
    }

    while (n_samples > num_clusters) {
        auto [cluster_a, cluster_b, min_dist] = find_closest_clusters(distances);
        for (int i = 0; i < labels.size(); ++i) {
            if (labels[i] == cluster_b) {
                labels[i] = cluster_a;
            }
        }
        update_distance_matrix(distances, cluster_a, cluster_b);
        --n_samples;
    }
}

std::vector<int> HierarchicalClustering::get_labels() {
    return labels;
}

double HierarchicalClustering::compute_distance(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
    return (a - b).norm();
}

std::tuple<int, int, double> HierarchicalClustering::find_closest_clusters(const Eigen::MatrixXd& distances) {
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

void HierarchicalClustering::update_distance_matrix(Eigen::MatrixXd& distances, int cluster_a, int cluster_b) {
    for (int i = 0; i < distances.rows(); ++i) {
        if (i != cluster_a && i != cluster_b) {
            distances(cluster_a, i) = distances(i, cluster_a) = std::min(distances(cluster_a, i), distances(cluster_b, i));
        }
    }
    distances.col(cluster_b).setConstant(std::numeric_limits<double>::max());
    distances.row(cluster_b).setConstant(std::numeric_limits<double>::max());
}
