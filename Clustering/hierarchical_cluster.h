#ifndef HIERARCHICALCLUSTER_H
#define HIERARCHICALCLUSTER_H

#include "cluster.h"
#include <vector>
#include <cmath>

class HierarchicalCluster : public Cluster {
    public:
        // Constructor
        HierarchicalCluster(size_t id) : Cluster(id) {}

        // Performs hierarchical clustering on the given data
        static std::vector<HierarchicalCluster> perform_clustering(const std::vector<std::vector<double>>& data, size_t num_clusters);

    private:
        // Calculates the Euclidean distance between two points
        static double euclidean_distance(const std::vector<double>& point1, const std::vector<double>& point2);

        // Merges two clusters
        static void merge_clusters(HierarchicalCluster& cluster1, HierarchicalCluster& cluster2);
};

#endif // HIERARCHICALCLUSTER_H
