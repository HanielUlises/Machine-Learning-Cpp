#include "logistic_regression.h"
#include <iostream>
#include <vector>

int main() {
    // Example data
    std::vector<std::vector<double>> X = {
        {1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0},
        {4.0, 5.0}, {5.0, 6.0}, {6.0, 7.0}
    };
    std::vector<int> y = {0, 0, 0, 1, 1, 1};

    // Logistic Regression model
    LogisticRegression model(0.01, 1000);

    // Train the model
    model.fit(X, y);

    // Make predictions
    std::vector<int> predictions = model.predict(X);

    // Output the predictions
    std::cout << "Predictions:" << std::endl;
    for (int pred : predictions) {
        std::cout << pred << std::endl;
    }

    return 0;
}
