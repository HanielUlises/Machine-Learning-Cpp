#ifndef HIERARCHICALCLUSTERING_H
#define HIERARCHICALCLUSTERING_H

#include <Eigen/Dense>
#include <vector>
#include <tuple>

class HierarchicalClustering {
public:
    HierarchicalClustering(int num_clusters);
    void fit(const Eigen::MatrixXd& data);
    std::vector<int> get_labels();

private:
    int num_clusters;
    std::vector<int> labels;
    double compute_distance(const Eigen::VectorXd& a, const Eigen::VectorXd& b);
    std::tuple<int, int, double> find_closest_clusters(const Eigen::MatrixXd& distances);
    void update_distance_matrix(Eigen::MatrixXd& distances, int cluster_a, int cluster_b);
};

#endif // HIERARCHICALCLUSTERING_H
