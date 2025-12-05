#ifndef PCA_H
#define PCA_H

#include <Eigen/Dense>

class PCA {
public:
    PCA(int num_components);
    void fit(const Eigen::MatrixXd& data);
    Eigen::MatrixXd transform(const Eigen::MatrixXd& data);
    Eigen::MatrixXd inverse_transform(const Eigen::MatrixXd& transformed_data);
    Eigen::MatrixXd get_components();
    Eigen::VectorXd get_mean();

private:
    int num_components;                // Number of principal components to retain
    Eigen::MatrixXd components;        // Matrix to store the principal components
    Eigen::VectorXd mean;              // Vector to store the mean of the data
};

#endif // PCA_H
