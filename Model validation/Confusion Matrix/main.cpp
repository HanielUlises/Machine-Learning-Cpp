#include <iostream>
#include <vector>
#include "KMeans.h"
#include "ConfusionMatrix.h"

int main() {
    // Example data: 2D points
    std::vector<std::vector<double>> data = {
        {1.0, 2.0}, {1.5, 1.8}, {5.0, 8.0}, {8.0, 8.0},
        {1.0, 0.6}, {9.0, 11.0}, {8.0, 2.0}, {10.0, 2.0},
        {9.0, 3.0}
    };

    // True labels (for demonstration purposes, we use some made-up labels)
    std::vector<int> true_labels = {0, 0, 1, 1, 0, 1, 1, 1, 1};

    // Instantiate KMeans with 2 clusters and fit the data
    KMeans kmeans(2, 100);
    kmeans.fit(data);

    // Predict the labels
    std::vector<int> predicted_labels = kmeans.predict(data);

    // Instantiate ConfusionMatrix with 2 classes and update it with predictions
    ConfusionMatrix cm(2);
    for (size_t i = 0; i < true_labels.size(); ++i) {
        cm.update(true_labels[i], predicted_labels[i]);
    }

    // Print the confusion matrix
    std::cout << "Confusion Matrix:" << std::endl;
    cm.print();

    return 0;
}
