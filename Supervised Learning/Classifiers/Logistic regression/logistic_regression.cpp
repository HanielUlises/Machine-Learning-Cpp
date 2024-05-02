#include "logistic_regression.h"
#include <cmath>
#include <numeric>

LogisticRegression::LogisticRegression(double lr, int iterations)
    : learningRate(lr), maxIterations(iterations) {
}

void LogisticRegression::fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
    size_t features = X[0].size();
    weights.resize(features, 0);

    for (int i = 0; i < maxIterations; ++i) {
        std::vector<double> grads = gradient(X, y);
        for (size_t j = 0; j < weights.size(); ++j) {
            weights[j] -= learningRate * grads[j];
        }
    }
}

std::vector<int> LogisticRegression::predict(const std::vector<std::vector<double>>& X) const {
    std::vector<int> predictions;
    for (const auto& x : X) {
        double prob = predict_proba(x);
        predictions.push_back(prob >= 0.5 ? 1 : 0);
    }
    return predictions;
}

double LogisticRegression::predict_proba(const std::vector<double>& x) const {
    double linear_model = std::inner_product(x.begin(), x.end(), weights.begin(), 0.0);
    return sigmoid(linear_model);
}

double LogisticRegression::sigmoid(double z) const {
    return 1 / (1 + exp(-z));
}

std::vector<double> LogisticRegression::gradient(const std::vector<std::vector<double>>& X, const std::vector<int>& y) const {
    std::vector<double> grads(weights.size(), 0.0);
    for (size_t i = 0; i < X.size(); ++i) {
        double predicted = predict_proba(X[i]);
        for (size_t j = 0; j < weights.size(); ++j) {
            grads[j] += (predicted - y[i]) * X[i][j];
        }
    }
    return grads;
}
