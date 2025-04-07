#include "hierarchical_cluster.h"

// Performs hierarchical clustering on the given data, returning the resulting clusters.
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
    // Initializes each data point as a separate cluster.
    for (size_t i = 0; i < data.size(); ++i) {
        HierarchicalCluster cluster(i);
        cluster.add_item(i);
        clusters.push_back(cluster);
    }

    // Continues merging clusters until the desired number of clusters is reached.
    while (clusters.size() > num_clusters) {
        size_t min_i = 0, min_j = 1;
        double min_distance = calculate_distance(clusters[0], clusters[1], data, distance_metric, linkage);

        // Finds the two closest clusters based on the specified distance metric and linkage method.
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

        // Merges the two closest clusters into one.
        merge_clusters(clusters[min_i], clusters[min_j]);
        clusters.erase(clusters.begin() + min_j); // Removes the merged cluster.
    }

    return clusters; // Returns the final clusters.
}

// Calculates the distance between two clusters based on the specified linkage method.
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

// Calculates the minimum distance between points in the two clusters (single linkage).
double HierarchicalCluster::single_linkage(const HierarchicalCluster& cluster1, const HierarchicalCluster& cluster2,
                                           const std::vector<std::vector<double>>& data, const Distance& distance_metric) {
    double min_distance = std::numeric_limits<double>::infinity();
    for (auto item1 : cluster1.data()) {
        for (auto item2 : cluster2.data()) {
            double distance = distance_metric(data[item1], data[item2]);
            if (distance < min_distance) {
                min_distance = distance; // Updates the minimum distance found.
            }
        }
    }
    return min_distance; // Returns the minimum distance between the clusters.
}

// Calculates the maximum distance between points in the two clusters (complete linkage).
double HierarchicalCluster::complete_linkage(const HierarchicalCluster& cluster1, const HierarchicalCluster& cluster2,
                                             const std::vector<std::vector<double>>& data, const Distance& distance_metric) {
    double max_distance = 0.0;
    for (auto item1 : cluster1.data()) {
        for (auto item2 : cluster2.data()) {
            double distance = distance_metric(data[item1], data[item2]);
            if (distance > max_distance) {
                max_distance = distance; // Updates the maximum distance found.
            }
        }
    }
    return max_distance; // Returns the maximum distance between the clusters.
}

// Calculates the average distance between all pairs of points in the two clusters (average linkage).
double HierarchicalCluster::average_linkage(const HierarchicalCluster& cluster1, const HierarchicalCluster& cluster2,
                                            const std::vector<std::vector<double>>& data, const Distance& distance_metric) {
    double total_distance = 0.0;
    size_t count = 0;
    for (auto item1 : cluster1.data()) {
        for (auto item2 : cluster2.data()) {
            total_distance += distance_metric(data[item1], data[item2]); // Accumulates total distance.
            ++count; // Counts the number of pairs.
        }
    }
    return count > 0 ? total_distance / count : 0.0; // Returns the average distance.
}

// Merges two clusters, adding the items of the second cluster to the first.
void HierarchicalCluster::merge_clusters(HierarchicalCluster& cluster1, HierarchicalCluster& cluster2) {
    for (auto item : cluster2.data()) {
        cluster1.add_item(item); // Adds each item from the second cluster to the first.
    }
}
