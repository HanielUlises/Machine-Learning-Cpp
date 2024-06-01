#include "Concept.h"
#include <cmath>
#include <stdexcept>
#include <numeric>

ConceptLearning::ConceptLearning(double lr, double reg) : learning_rate(lr), regularization_strength(reg) {}

void ConceptLearning::train(const std::vector<std::vector<double>>& examples, const std::vector<int>& labels, int epochs) {
    if (examples.empty() || examples[0].empty()) throw std::invalid_argument("Empty training data");
    weights.resize(examples[0].size(), 0.0);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < examples.size(); ++i) {
            updateWeights(examples[i], labels[i]);
        }
    }
}

double ConceptLearning::sigmoid(double z) const {
    return 1.0 / (1.0 + exp(-z));
}

void ConceptLearning::updateWeights(const std::vector<double>& example, int label) {
    double predicted = sigmoid(std::inner_product(weights.begin(), weights.end(), example.begin(), 0.0));
    double error = label - predicted;
    for (size_t i = 0; i < weights.size(); ++i) {
        // Updating with regularization
        weights[i] += learning_rate * (error * example[i] - regularization_strength * weights[i]);
    }
}

int ConceptLearning::predict(const std::vector<double>& example) {
    double sum = std::inner_product(weights.begin(), weights.end(), example.begin(), 0.0);
    return sigmoid(sum) > 0.5 ? 1 : 0;
}

double ConceptLearning::accuracy(const std::vector<std::vector<double>>& test_examples, const std::vector<int>& test_labels) {
    int correct = 0;
    for (size_t i = 0; i < test_examples.size(); ++i) {
        if (predict(test_examples[i]) == test_labels[i]) {
            correct++;
        }
    }
    return static_cast<double>(correct) / test_examples.size();
}

void ConceptLearning::loadDataFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Failed to open file");

    std::vector<double> example;
    int label;
    while (file >> label) {
        example.clear();
        double feature;
        while (file.peek() != '\n' && file >> feature) {
            example.push_back(feature);
        }
        train({example}, {label}, 1); 
    }
    file.close();
}
