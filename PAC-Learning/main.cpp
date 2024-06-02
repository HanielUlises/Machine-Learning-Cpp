#include "Concept.h"
#include <iostream>

int main() {
    ConceptLearning model(0.01, 0.1);

    std::vector<std::vector<double>> training_examples = {
        {2.0, 3.0}, {1.0, 1.0}, {2.5, 3.5}, {3.0, 2.0}
    };
    std::vector<int> training_labels = {1, 0, 1, 0};

    // Train the model
    model.train(training_examples, training_labels, 1000);

    // Trained model to a file
    std::string model_filename = "model.txt";
    model.saveModelToFile(model_filename);
    std::cout << "Model saved to " << model_filename << "\n";

    // Loading the model from the file
    ConceptLearning loaded_model;
    loaded_model.loadModelFromFile(model_filename);
    std::cout << "Model loaded from " << model_filename << "\n";

    // Test data: examples and corresponding labels
    std::vector<std::vector<double>> test_examples = {
        {2.0, 3.0}, {1.0, 1.0}, {2.5, 3.5}, {3.0, 2.0}
    };
    std::vector<int> test_labels = {1, 0, 1, 0};

    // Calculate the accuracy of the loaded model on the test data
    double test_accuracy = loaded_model.accuracy(test_examples, test_labels);
    std::cout << "Test Accuracy: " << test_accuracy * 100 << "%\n";

    return 0;
}
