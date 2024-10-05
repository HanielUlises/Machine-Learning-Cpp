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

// The training function iteratively updates the weights of the logistic regression model.
// It processes each example in the dataset for a specified number of epochs, adjusting weights
// based on the error between predicted and actual labels.
void ConceptLearning::train(const std::vector<std::vector<double>>& examples, const std::vector<int>& labels, int epochs) {
    if (examples.empty() || examples[0].empty()) throw std::invalid_argument("Empty training data");
    weights.resize(examples[0].size(), 0.0);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < examples.size(); ++i) {
            update_weights(examples[i], labels[i]);
        }
    }
}

// The sigmoid function applies the logistic transformation to the input,
// converting any real-valued number into the range of (0, 1), which is suitable for probability interpretation.
double ConceptLearning::sigmoid(double z) const {
    return 1.0 / (1.0 + exp(-z));
}

// The weight updating process uses the calculated prediction error to adjust the weights
// according to the learning rate and includes L2 regularization to prevent overfitting.
void ConceptLearning::update_weights(const std::vector<double>& example, int label) {
    double z = std::inner_product(weights.begin(), weights.end(), example.begin(), 0.0);
    double predicted = sigmoid(z);
    double error = label - predicted;

    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] += learning_rate * (error * example[i] - regularization_strength * weights[i]);
    }
}

// The predict function calculates the label of a new example by applying the trained model,
// determining if the input features correspond to a positive or negative class based on the learned weights.
int ConceptLearning::predict(const std::vector<double>& example) {
    double z = std::inner_product(weights.begin(), weights.end(), example.begin(), 0.0);
    return sigmoid(z) > 0.5 ? 1 : 0;
}

// The accuracy function evaluates the performance of the model on a test dataset,
// calculating the proportion of correctly predicted labels out of the total examples.
double ConceptLearning::accuracy(const std::vector<std::vector<double>>& test_examples, const std::vector<int>& test_labels) {
    int correct = 0;  
    for (size_t i = 0; i < test_examples.size(); ++i) {
        if (predict(test_examples[i]) == test_labels[i]) {
            correct++;
        }
    }
    return static_cast<double>(correct) / test_examples.size();
}

// The load data function reads examples and their corresponding labels from a file,
// updating the model with each example sequentially to facilitate training from external datasets.
void ConceptLearning::load_data_from_file(const std::string& filename) {
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

// The save model function writes the model's weights to a specified file,
// allowing for future retrieval and use without the need to retrain.
void ConceptLearning::save_model_to_file(const std::string& filename) const {
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

// The load model function retrieves the model's weights from a specified file,
// restoring the state of the model for continued use or evaluation.
void ConceptLearning::load_model_from_file(const std::string& filename) {
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

// The calculate generalization error function provides an estimate of the model's
// expected error on unseen data based on the sample size and confidence level,
double ConceptLearning::calculate_generalization_error(double sample_size, double delta) const {
    double h_size = weights.size();
    return sqrt((1.0 / (2 * sample_size)) * (h_size * log((2 * sample_size) / h_size) + log(1 / delta)));
}
