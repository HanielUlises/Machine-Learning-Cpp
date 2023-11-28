#include "knn.h"
#include <iostream>
#include <fstream>
#include <sstream>

int main() {
    std::vector<Point> trainingData;

    std::ifstream file("training_data.txt");
    if (file.is_open()) {
        double x, y;
        while (file >> x >> y) {
            trainingData.push_back({x, y});
        }
        file.close();
    } else {
        std::cerr << "Unable to open file." << std::endl;
        return 1;
    }

    double testX, testY;

    std::cout << "Reading values for the new point" << std::endl;

    std::cout << "Introduce the x value: ";
    std::cin >> testX;
    std::cout << "Introduce the y value: ";
    std::cin >> testY;
    std::cout << std::endl;

    Point testPoint = {testX, testY};

    int k;
    std::cout << "Enter the value of k: ";
    std::cin >> k;

    KNN knnClassifier(trainingData);
    Point result = knnClassifier.classify(testPoint, k);

    std::cout << "The new point is classified as (" << result.x << ", " << result.y << ")" << std::endl;

    return 0;
}