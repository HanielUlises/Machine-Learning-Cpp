#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "linear_regression.h"
#include "Plotting.h"

std::pair<std::vector<double>, std::vector<double>> read_csv(const std::string& filename) {
    std::ifstream file(filename);
    std::string line, value;
    std::vector<double> X, y;
    
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::getline(ss, value, ',');
        X.push_back(std::stod(value));
        std::getline(ss, value, ',');
        y.push_back(std::stod(value));
    }
    
    return {X, y};
}

int main() {
    auto [X, y] = read_csv("data.csv");
    
    LinearRegression lr;
    lr.fit(X, y);
    
    // Make predictions
    std::vector<double> predictions = lr.predict(X);
    
    std::cout << "Predictions: ";
    for (const auto& pred : predictions) {
        std::cout << pred << " ";
    }
    std::cout << std::endl;
    
    plot_regression_line(X, y, predictions, "Regression Line from CSV Data");
    
    return 0;
}
