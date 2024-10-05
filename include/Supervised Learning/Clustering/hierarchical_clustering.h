#ifndef HIERARCHICAL_CLUSTERING_H
#define HIERARCHICAL_CLUSTERING_H

#include <Eigen/Dense>
#include <vector>
#include <tuple>

class HierarchicalClustering {
public:
    // Constructor initializes the clustering object with a specified number of clusters.
    explicit HierarchicalClustering(int num_clusters);

    // Fit the hierarchical clustering model to the provided data matrix.
    void fit(const Eigen::MatrixXd& data);

    // Retrieve the cluster labels assigned to each data point.
    std::vector<int> get_labels() const;

private:
    int num_clusters_;  // Number of desired clusters
    std::vector<int> labels_;  // Cluster labels for each data point

    // Compute the distance between two data points represented as vectors.
    double compute_distance(const Eigen::VectorXd& point_a, const Eigen::VectorXd& point_b) const;

    // Identify the closest clusters based on the distance matrix.
    std::tuple<int, int, double> find_closest_clusters(const Eigen::MatrixXd& distance_matrix) const;

    // Update the distance matrix after merging two clusters.
    void update_distance_matrix(Eigen::MatrixXd& distance_matrix, int cluster_a, int cluster_b) noexcept;
};

#endif // HIERARCHICAL_CLUSTERING_H
