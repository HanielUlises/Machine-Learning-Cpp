#include <iostream>
#include "PCA.h"

int main() {
    Eigen::MatrixXd data(5, 3);
    data << 2.5, 2.4, 2.8,
            0.5, 0.7, 0.6,
            2.2, 2.9, 2.7,
            1.9, 2.2, 2.5,
            3.1, 3.0, 3.3;

    PCA pca(2);
    pca.fit(data);
    Eigen::MatrixXd transformed_data = pca.transform(data);

    std::cout << "Original Data:\n" << data << std::endl;
    std::cout << "Transformed Data:\n" << transformed_data << std::endl;
    std::cout << "Reconstructed Data:\n" << pca.inverse_transform(transformed_data) << std::endl;

    return 0;
}