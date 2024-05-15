#include "hierarchical_cluster.h"

// Performs hierarchical clustering on the given data
std::vector<HierarchicalCluster> HierarchicalCluster::perform_clustering(const std::vector<std::vector<double>>& data, size_t num_clusters) {
    std::vector<HierarchicalCluster> clusters;
    // Initialize each data point as a separate cluster
    for (size_t i = 0; i < data.size(); ++i) {
        HierarchicalCluster cluster(i);
        cluster.add_item(i);
        clusters.push_back(cluster);
    }

    while (clusters.size() > num_clusters) {
        size_t min_i = 0, min_j = 1;
        double min_distance = euclidean_distance(data[clusters[0].data().begin() - data.begin()], data[clusters[1].data().begin() - data.begin()]);

        // Find the two closest clusters
        for (size_t i = 0; i < clusters.size(); ++i) {
            for (size_t j = i + 1; j < clusters.size(); ++j) {
                double distance = euclidean_distance(data[*clusters[i].data().begin()], data[*clusters[j].data().begin()]);
                if (distance < min_distance) {
                    min_distance = distance;
                    min_i = i;
                    min_j = j;
                }
            }
        }

        // Merge the two closest clusters
        merge_clusters(clusters[min_i], clusters[min_j]);
        clusters.erase(clusters.begin() + min_j);
    }

    return clusters;
}

// Calculates the Euclidean distance between two points
double HierarchicalCluster::euclidean_distance(const std::vector<double>& point1, const std::vector<double>& point2) {
    double sum = 0;
    for (size_t i = 0; i < point1.size(); ++i) {
        sum += std::pow(point1[i] - point2[i], 2);
    }
    return std::sqrt(sum);
}

// Merges two clusters
void HierarchicalCluster::merge_clusters(HierarchicalCluster& cluster1, HierarchicalCluster& cluster2) {
    for (auto item : cluster2.data()) {
        cluster1.add_item(item);
    }
}
