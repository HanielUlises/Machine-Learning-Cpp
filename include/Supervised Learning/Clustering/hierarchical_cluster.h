#ifndef HIERARCHICALCLUSTER_H
#define HIERARCHICALCLUSTER_H

#include "cluster.h"
#include "distance.h"
#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>

// Represents a hierarchical cluster, extending the base Cluster class with size_t items.
class HierarchicalCluster : public Cluster<size_t> {
public:
    // Constructs a hierarchical cluster with a specified identifier.
    HierarchicalCluster(size_t id) : Cluster<size_t>(id) {}

    // Constructs a hierarchical cluster with a specified identifier and initializes it with items.
    HierarchicalCluster(size_t id, const std::vector<size_t>& items) : Cluster<size_t>(id, items) {}

    // Performs hierarchical clustering on a given dataset, generating a specified number of clusters using the provided distance metric and linkage method.
    static std::vector<HierarchicalCluster> perform_clustering(
        const std::vector<std::vector<double>>& data, size_t num_clusters,
        const Distance& distance_metric, const std::string& linkage);

private:
    // Calculates the distance between two clusters using a specific distance metric and linkage method.
    static double calculate_distance(const HierarchicalCluster& cluster1, const HierarchicalCluster& cluster2,
                                     const std::vector<std::vector<double>>& data, const Distance& distance_metric,
                                     const std::string& linkage);

    // Merges two clusters into one, updating the state of the first cluster.
    static void merge_clusters(HierarchicalCluster& cluster1, HierarchicalCluster& cluster2);

    // Computes distance based on the single linkage method, identifying the shortest distance between two clusters.
    static double single_linkage(const HierarchicalCluster& cluster1, const HierarchicalCluster& cluster2,
                                 const std::vector<std::vector<double>>& data, const Distance& distance_metric);

    // Computes distance based on the complete linkage method, identifying the longest distance between two clusters.
    static double complete_linkage(const HierarchicalCluster& cluster1, const HierarchicalCluster& cluster2,
                                   const std::vector<std::vector<double>>& data, const Distance& distance_metric);

    // Computes distance based on the average linkage method, identifying the average distance between all pairs of points in two clusters.
    static double average_linkage(const HierarchicalCluster& cluster1, const HierarchicalCluster& cluster2,
                                  const std::vector<std::vector<double>>& data, const Distance& distance_metric);
};

#endif // HIERARCHICALCLUSTER_H
