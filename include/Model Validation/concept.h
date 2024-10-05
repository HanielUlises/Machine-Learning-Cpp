#ifndef CONCEPT_H
#define CONCEPT_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>

class ConceptLearning {
public:
    ConceptLearning(double lr = 0.01, double reg = 0.1);
    
    // The train function iteratively adjusts the model weights based on the training examples and their corresponding labels.
    void train(const std::vector<std::vector<double>>& examples, const std::vector<int>& labels, int epochs);
    
    // The predict function determines the label of a new example using the trained model weights.
    int predict(const std::vector<double>& example);
    
    // The accuracy function evaluates the model's performance by calculating the proportion of correctly predicted examples.
    double accuracy(const std::vector<std::vector<double>>& test_examples, const std::vector<int>& test_labels);
    
    // The load_data_from_file function reads training data from a specified file, facilitating model training with external datasets.
    void load_data_from_file(const std::string& filename);
    
    // The save_model_to_file function stores the current model weights in a specified file for future use.
    void save_model_to_file(const std::string& filename) const;
    
    // The load_model_from_file function retrieves model weights from a specified file to restore the model's state.
    void load_model_from_file(const std::string& filename);
    
    // The calculate_generalization_error function estimates the expected error of the model on unseen data based on sample size and confidence level.
    double calculate_generalization_error(double sample_size, double delta) const;

private:
    std::vector<double> weights; // Logistic regression weights
    double learning_rate; // The learning rate for weight updates
    double regularization_strength; // The strength of L2 regularization to prevent overfitting
    double sigmoid(double z) const; // The sigmoid activation function
    void update_weights(const std::vector<double>& example, int label); // Updates model weights based on a single example
};

#endif
