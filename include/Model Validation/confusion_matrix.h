#ifndef CONFUSIONMATRIX_H
#define CONFUSIONMATRIX_H

#include <vector>

// Class representing a confusion matrix, which is used to evaluate the performance of a classification model.
// It tracks the true labels against the predicted labels for a specified number of classes.
class ConfusionMatrix {
public:
    // Initializes the confusion matrix with the given number of classes.
    ConfusionMatrix(int num_classes);

    void update(int true_label, int predicted_label);
    void print() const;

private:
    std::vector<std::vector<int>> matrix; // true vs. predicted labels.
    int num_classes;                       // Number of classes in the classification problem.
};

#endif // CONFUSIONMATRIX_H
