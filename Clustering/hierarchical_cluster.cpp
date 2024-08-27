#include "hierarchical_cluster.h"

// Performs hierarchical clustering on the given data
std::vector<HierarchicalCluster> HierarchicalCluster::perform_clustering(
    const std::vector<std::vector<double>>& data, size_t num_clusters,
    const Distance& distance_metric, const std::string& linkage) {

    if (data.empty()) {
        throw std::invalid_argument("Data is empty.");
    }

    if (num_clusters > data.size()) {
        throw std::invalid_argument("Number of clusters cannot be more than number of data points.");
    }

    std::vector<HierarchicalCluster> clusters;
    // Initialization
    // Each data point as a separate cluster
    for (size_t i = 0; i < data.size(); ++i) {
        HierarchicalCluster cluster(i);
        cluster.add_item(i);
        clusters.push_back(cluster);
    }

    while (clusters.size() > num_clusters) {
        size_t min_i = 0, min_j = 1;
        double min_distance = calculate_distance(clusters[0], clusters[1], data, distance_metric, linkage);

        // Find the two closest clusters
        for (size_t i = 0; i < clusters.size(); ++i) {
            for (size_t j = i + 1; j < clusters.size(); ++j) {
                double distance = calculate_distance(clusters[i], clusters[j], data, distance_metric, linkage);
                if (distance < min_distance) {
                    min_distance = distance;
                    min_i = i;
                    min_j = j;
                }
            }
        }

        // Merging the two closest clusters
        merge_clusters(clusters[min_i], clusters[min_j]);
        clusters.erase(clusters.begin() + min_j);
    }

    return clusters;
}

// Calculates the distance between two clusters based on the specified linkage method
double HierarchicalCluster::calculate_distance(const HierarchicalCluster& cluster1, const HierarchicalCluster& cluster2,
                                               const std::vector<std::vector<double>>& data, const Distance& distance_metric,
                                               const std::string& linkage) {
    if (linkage == "single") {
        return single_linkage(cluster1, cluster2, data, distance_metric);
    } else if (linkage == "complete") {
        return complete_linkage(cluster1, cluster2, data, distance_metric);
    } else if (linkage == "average") {
        return average_linkage(cluster1, cluster2, data, distance_metric);
    } else {
        throw std::invalid_argument("Unknown linkage method.");
    }
}

// Single linkage: Minimum distance between points in the two clusters
double HierarchicalCluster::single_linkage(const HierarchicalCluster& cluster1, const HierarchicalCluster& cluster2,
                                           const std::vector<std::vector<double>>& data, const Distance& distance_metric) {
    double min_distance = std::numeric_limits<double>::infinity();
    for (auto item1 : cluster1.data()) {
        for (auto item2 : cluster2.data()) {
            double distance = distance_metric(data[item1], data[item2]);
            if (distance < min_distance) {
                min_distance = distance;
            }
        }
    }
    return min_distance;
}

// Complete linkage: Maximum distance between points in the two clusters
double HierarchicalCluster::complete_linkage(const HierarchicalCluster& cluster1, const HierarchicalCluster& cluster2,
                                             const std::vector<std::vector<double>>& data, const Distance& distance_metric) {
    double max_distance = 0.0;
    for (auto item1 : cluster1.data()) {
        for (auto item2 : cluster2.data()) {
            double distance = distance_metric(data[item1], data[item2]);
            if (distance > max_distance) {
                max_distance = distance;
            }
        }
    }
    return max_distance;
}

// Average linkage: Average distance between all pairs of points in the two clusters
double HierarchicalCluster::average_linkage(const HierarchicalCluster& cluster1, const HierarchicalCluster& cluster2,
                                            const std::vector<std::vector<double>>& data, const Distance& distance_metric) {
    double total_distance = 0.0;
    size_t count = 0;
    for (auto item1 : cluster1.data()) {
        for (auto item2 : cluster2.data()) {
            total_distance += distance_metric(data[item1], data[item2]);
            ++count;
        }
    }
    return count > 0 ? total_distance / count : 0.0;
}

// Merges two clusters
void HierarchicalCluster::merge_clusters(HierarchicalCluster& cluster1, HierarchicalCluster& cluster2) {
    for (auto item : cluster2.data()) {
        cluster1.add_item(item);
    }
}

