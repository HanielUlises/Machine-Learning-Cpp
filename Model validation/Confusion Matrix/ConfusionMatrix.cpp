#include "ConfusionMatrix.h"
#include <iostream>
#include <iomanip>

ConfusionMatrix::ConfusionMatrix(int num_classes) : num_classes(num_classes) {
    matrix.resize(num_classes, std::vector<int>(num_classes, 0));
}

void ConfusionMatrix::update(int true_label, int predicted_label) {
    matrix[true_label][predicted_label]++;
}

void ConfusionMatrix::print() const {
    for (const auto& row : matrix) {
        for (int val : row) {
            std::cout << std::setw(5) << val << " ";
        }
        std::cout << std::endl;
    }
}
