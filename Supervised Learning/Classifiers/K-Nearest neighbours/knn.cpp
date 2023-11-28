#include "knn.h"

KNN::KNN (const std::vector<Point>& trainingData) : trainingData(trainingData){}

double KNN::calculateDistance(const Point& p1, const Point& p2){
    return std::sqrt((std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2)));
}

Point KNN::classify (const Point& p, int k){
    std::vector<std::pair<double, int>> distances;

    for(int i = 0; i < trainingData.size(); i++){
        double distance = calculateDistance(p, trainingData[i]);
        distances.emplace_back(distance, i);
    }

    std::sort(distances.begin(), distances.end());

    double sumX = 0.0f;
    double sumY = 0.0f;
    for(int i = 0; i < k; i++){
        sumX += trainingData[distances[i].second].x;
        sumY += trainingData[distances[i].second].y;
    }

    Point result;
    result.x = sumX/k;
    result.y = sumY/k;

    return result;
}