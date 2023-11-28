// main.cpp

#include "minimumDistanceClassifier.h"
#include <fstream>
#include <sstream>

int main() {
    std::vector<Instance> trainingData;

    // Retrieving raining data from file
    std::ifstream file("training_data.txt");
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            Instance instance;
            iss >> instance.features[0] >> instance.features[1] >> instance.label;
            trainingData.push_back(instance);
        }
        file.close();
    } else {
        std::cerr << "Unable to open file." << std::endl;
        return 1;
    }

    // New instance to be tested
    double x, y;
    int label;

    std::cout << "Reading values for the new instance" << std::endl;

    std::cout<<"Introduce the x value: "; std::cin >> x;
    std::cout<<"Introduce the y value: "; std::cin >> y;
    std::cout<<"Introduce the label: "; std::cin >> label; std::cout<<std::endl;

    Instance newInstance = {{x, y}, label};

    int result = classify(trainingData, newInstance);

    if (result != -1) {
        std::cout << "The new instance is classified as class " << result << std::endl;
    }

    return 0;
}