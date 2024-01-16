#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <random>
#include <chrono>

class KMeans {
public:
    KMeans(int k, const std::vector<Point>& data) : k(k), data(data) {
        initializeCentroids();
    }

    void run(int iterations) {
        for (int i = 0; i < iterations; ++i) {
            assignClusters();
            updateCentroids();
        }
    }

    const std::vector<Point>& getCentroids() const {
        return centroids;
    }

private:
    struct Point {
        double x, y;
    };

    int k;
    std::vector<Point> centroids;
    std::vector<Point> data;
    std::vector<int> labels;

    void initializeCentroids() {
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine engine(seed);

        std::shuffle(data.begin(), data.end(), engine);

        centroids.assign(data.begin(), data.begin() + k);
    }

    double calculateDistance(const Point& p1, const Point& p2) {
        return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
    }

    void assignClusters() {
        labels.resize(data.size());
        for (int i = 0; i < data.size(); ++i) {
            double minDist = std::numeric_limits<double>::max();
            int clusterIndex = 0;
            for (int j = 0; j < k; ++j) {
                double dist = calculateDistance(data[i], centroids[j]);
                if (dist < minDist) {
                    minDist = dist;
                    clusterIndex = j;
                }
            }
            labels[i] = clusterIndex;
        }
    }

    void updateCentroids() {
        std::vector<int> count(k, 0);
        std::vector<Point> newCentroids(k, {0, 0});
        for (int i = 0; i < data.size(); ++i) {
            newCentroids[labels[i]].x += data[i].x;
            newCentroids[labels[i]].y += data[i].y;
            count[labels[i]]++;
        }

        for (int i = 0; i < k; ++i) {
            if (count[i] != 0) {
                newCentroids[i].x /= count[i];
                newCentroids[i].y /= count[i];
            }
        }

        centroids = newCentroids;
    }
};
