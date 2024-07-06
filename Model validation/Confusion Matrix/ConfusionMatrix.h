#ifndef CONFUSIONMATRIX_H
#define CONFUSIONMATRIX_H

#include <vector>

class ConfusionMatrix {
public:
    ConfusionMatrix(int num_classes);
    void update(int true_label, int predicted_label);
    void print() const;

private:
    std::vector<std::vector<int>> matrix;
    int num_classes;
};

#endif // CONFUSIONMATRIX_H
