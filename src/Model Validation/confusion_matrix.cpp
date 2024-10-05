#include "ConfusionMatrix.h"
#include <iostream>
#include <iomanip>

// Constructor initializes the confusion matrix with the specified number of classes.
// The matrix is resized to a 2D vector, where each element is initialized to zero.
ConfusionMatrix::ConfusionMatrix(int num_classes) : num_classes(num_classes) {
    matrix.resize(num_classes, std::vector<int>(num_classes, 0));
}

// Updates the confusion matrix by incrementing the count for the true and predicted labels.
// This keeps track of how many times a particular prediction was made for a given true label.
void ConfusionMatrix::update(int true_label, int predicted_label) {
    matrix[true_label][predicted_label]++;
}

// Prints the confusion matrix to the console in a formatted manner.
// Each row corresponds to the true label, while each column corresponds to the predicted label.
void ConfusionMatrix::print() const {
    for (const auto& row : matrix) {
        for (int val : row) {
            std::cout << std::setw(5) << val << " "; 
        }
        std::cout << std::endl; 
    }
}
