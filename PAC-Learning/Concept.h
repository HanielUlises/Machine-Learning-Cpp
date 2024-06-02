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
    void train(const std::vector<std::vector<double>>& examples, const std::vector<int>& labels, int epochs);
    int predict(const std::vector<double>& example);
    double accuracy(const std::vector<std::vector<double>>& test_examples, const std::vector<int>& test_labels);
    void loadDataFromFile(const std::string& filename);
    void saveModelToFile(const std::string& filename) const;
    void loadModelFromFile(const std::string& filename);

    double calculateGeneralizationError(double sample_size, double delta) const;

private:
    std::vector<double> weights; // Logistic regression weights
    double learning_rate;
    double regularization_strength;
    double sigmoid(double z) const;
    void updateWeights(const std::vector<double>& example, int label);
};

#endif
