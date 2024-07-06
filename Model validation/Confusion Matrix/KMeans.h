#ifndef KMEANS_H
#define KMEANS_H

#include <vector>

class KMeans {
public:
    KMeans(int k, int max_iters);
    void fit(const std::vector<std::vector<double>>& data);
    std::vector<int> predict(const std::vector<std::vector<double>>& data);
    std::vector<std::vector<double>> getCentroids() const;

private:
    int k;
    int max_iters;
    std::vector<std::vector<double>> centroids;
    int closestCentroid(const std::vector<double>& point);
    std::vector<double> computeCentroid(const std::vector<std::vector<double>>& points);
};

#endif // KMEANS_H
