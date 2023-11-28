#include "minimumDistanceClassifier.h"

double calculateDistance(const Instance& instance1, const Instance& instance2) {
    double distance = 0.0f;

    for (size_t i = 0; i < 2; i++) {
        distance += std::pow(instance1.features[i] - instance2.features[i], 2);
    }
    return std::sqrt(distance);
}

int classify(const std::vector<Instance>& trainingData, const Instance& newInstance) {
    if (trainingData.empty()) {
        std::cerr << "There's no training data :(" << std::endl;
        return -1;
    }

    int minDistanceIndex = 0;
    double minDistance = calculateDistance(trainingData[0], newInstance);

    for (size_t i = 1; i < trainingData.size(); i++) {
        double distance = calculateDistance(trainingData[i], newInstance);
        if (distance < minDistance) {
            minDistance = distance;
            minDistanceIndex = static_cast<int>(i);
        }
    }
    return trainingData[minDistanceIndex].label;
}