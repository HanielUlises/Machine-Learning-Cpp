#include "Concept.h"
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <fstream>

ConceptLearning::ConceptLearning(double lr, double reg) : learning_rate(lr), regularization_strength(reg) {
    weights = {};
}

// Training the logistic regression model over a specified number of epochs
void ConceptLearning::train(const std::vector<std::vector<double>>& examples, const std::vector<int>& labels, int epochs) {
    // Non-empty training data
    if (examples.empty() || examples[0].empty()) throw std::invalid_argument("Empty training data");
    // Initialization of weights with zero values
    weights.resize(examples[0].size(), 0.0);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < examples.size(); ++i) {
            updateWeights(examples[i], labels[i]);
        }
    }
}

// Sigmoid activation function, computes 1 / (1 + exp(-z))
double ConceptLearning::sigmoid(double z) const {
    return 1.0 / (1.0 + exp(-z));
}

// Update model weights using the learning rate, regularization, and error calculation
void ConceptLearning::updateWeights(const std::vector<double>& example, int label) {
    // Calculate the dot product of weights and input features
    double z = std::inner_product(weights.begin(), weights.end(), example.begin(), 0.0);
    // Predict probability using the sigmoid function
    double predicted = sigmoid(z);
    // Compute error as difference between actual label and predicted probability
    double error = label - predicted;

    // Update each weight
    for (size_t i = 0; i < weights.size(); ++i) {
        // Apply gradient descent with L2 regularization
        weights[i] += learning_rate * (error * example[i] - regularization_strength * weights[i]);
    }
}

// Predict the label of a given example, returning 1 for positive class and 0 for negative class
int ConceptLearning::predict(const std::vector<double>& example) {
    // Compute the dot product of weights and input features
    double z = std::inner_product(weights.begin(), weights.end(), example.begin(), 0.0);
    // Return 1 if the sigmoid of z is greater than 0.5, else return 0
    return sigmoid(z) > 0.5 ? 1 : 0;
}

// Calculate the accuracy of the model on a test dataset
double ConceptLearning::accuracy(const std::vector<std::vector<double>>& test_examples, const std::vector<int>& test_labels) {
    // Count of correctly predicted samples
    int correct = 0;  
    // Iterate through each test example
    for (size_t i = 0; i < test_examples.size(); ++i) {
        // Increment correct count if the prediction matches the label
        if (predict(test_examples[i]) == test_labels[i]) {
            correct++;
        }
    }
    // Proportion of correctly predicted examples
    return static_cast<double>(correct) / test_examples.size();
}

// Load data from a file and train the model
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

// Save model weights to a file
void ConceptLearning::saveModelToFile(const std::string& filename) const {
    std::ofstream file(filename);
    if (file.is_open()) {
        for (const auto& weight : weights) {
            file << weight << "\n";
        }
        file.close();
    } else {
        std::cerr << "Unable to open file for saving model\n";
    }
}

// Load model weights from a file
void ConceptLearning::loadModelFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (file.is_open()) {
        weights.clear();
        std::string line;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            double weight;
            ss >> weight;
            weights.push_back(weight);
        }
        file.close();
    } else {
        std::cerr << "Unable to open file for loading model\n";
    }
}

// Calculate the generalization error bound
double ConceptLearning::calculateGeneralizationError(double sample_size, double delta) const {
    double h_size = weights.size();
    return sqrt((1.0 / (2 * sample_size)) * (h_size * log((2 * sample_size) / h_size) + log(1 / delta)));
}
