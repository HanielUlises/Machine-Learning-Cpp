#ifndef HIERARCHICALCLUSTER_H
#define HIERARCHICALCLUSTER_H

#include "cluster.h"
#include "distance.h"
#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>

class HierarchicalCluster : public Cluster<size_t> {
public:
    HierarchicalCluster(size_t id) : Cluster<size_t>(id) {}
    HierarchicalCluster(size_t id, const std::vector<size_t>& items) : Cluster<size_t>(id, items) {}

    // Perform hierarchical clustering with different linkage methods
    static std::vector<HierarchicalCluster> perform_clustering(
        const std::vector<std::vector<double>>& data, size_t num_clusters,
        const Distance& distance_metric, const std::string& linkage);

private:
    // Calculates the distance between two clusters based on a distance metric
    static double calculate_distance(const HierarchicalCluster& cluster1, const HierarchicalCluster& cluster2,
                                     const std::vector<std::vector<double>>& data, const Distance& distance_metric,
                                     const std::string& linkage);

    // Merge two clusters
    static void merge_clusters(HierarchicalCluster& cluster1, HierarchicalCluster& cluster2);

    // Distance calculations for different linkage methods
    static double single_linkage(const HierarchicalCluster& cluster1, const HierarchicalCluster& cluster2,
                                 const std::vector<std::vector<double>>& data, const Distance& distance_metric);

    static double complete_linkage(const HierarchicalCluster& cluster1, const HierarchicalCluster& cluster2,
                                   const std::vector<std::vector<double>>& data, const Distance& distance_metric);

    static double average_linkage(const HierarchicalCluster& cluster1, const HierarchicalCluster& cluster2,
                                  const std::vector<std::vector<double>>& data, const Distance& distance_metric);
};

#endif // HIERARCHICALCLUSTER_H
