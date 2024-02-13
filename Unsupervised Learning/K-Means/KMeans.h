#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <random>
#include <chrono>

// A point in R2 
struct Point {
    double x, y;        
};

class KMeans {
public:
    std::vector<Point> centroids;  // Centroids of clusters
    std::vector<Point> data;       // Data points

    KMeans(int k, const std::vector<Point>& data) : k(k), data(data) {
        initializeCentroids();  
    }

    void run(int iterations) {
        // Run KMeans clustering algorithm for a specified number of iterations
        for (int i = 0; i < iterations; ++i) {
            assignClusters();      // Assign data points to clusters
            updateCentroids();     // Update centroids based on assigned clusters
        }
    }

    const std::vector<Point>& getCentroids() const {
        return centroids;         // Return the centroids
    }

private:
    int k;                        // Number of clusters
    std::vector<int> labels;      // Cluster labels for each data point

    void initializeCentroids() {
        // Initialize centroids by randomly selecting data points
        centroids.clear();        // Clear existing centroids
        if (data.empty()) return; // Return if data is empty

        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count(); // Generate seed
        std::default_random_engine engine(seed); // Random engine with seed

        std::shuffle(data.begin(), data.end(), engine); // Shuffle data points
        centroids.assign(data.begin(), data.begin() + k); // Assign first k data points as centroids
    }

    double calculateDistance(const Point& p1, const Point& p2) {
        // Calculate Euclidean distance between two points
        return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
    }

    void assignClusters() {
        // Assign each data point to the nearest centroid
        labels.resize(data.size()); // Resize labels vector
        for (int i = 0; i < data.size(); ++i) {
            double minDist = std::numeric_limits<double>::max(); // Initialize minimum distance
            int clusterIndex = 0; // Initialize cluster index
            for (int j = 0; j < k; ++j) {
                double dist = calculateDistance(data[i], centroids[j]); // Calculate distance to centroid j
                if (dist < minDist) { // If distance is smaller than minimum
                    minDist = dist;   // Update minimum distance
                    clusterIndex = j; // Update cluster index
                }
            }
            labels[i] = clusterIndex; // Assign cluster index to data point
        }
    }

    void updateCentroids() {
        // Update centroids based on assigned clusters
        std::vector<int> count(k, 0);       // Vector to store count of data points in each cluster
        std::vector<Point> newCentroids(k, {0, 0}); // Vector to store new centroids
        for (int i = 0; i < data.size(); ++i) {
            newCentroids[labels[i]].x += data[i].x; // Accumulate x-coordinate
            newCentroids[labels[i]].y += data[i].y; // Accumulate y-coordinate
            count[labels[i]]++;             // Increment count for the cluster
        }

        for (int i = 0; i < k; ++i) {
            if (count[i] != 0) {      
                newCentroids[i].x /= count[i]; // Mean x-coordinate
                newCentroids[i].y /= count[i]; // Mean y-coordinate
            }
        }

        centroids = newCentroids; // Update centroids
    }
};