#include "PCA.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

PCA::PCA(int num_components) 
    : num_components(num_components) {}

void PCA::fit(const Eigen::MatrixXd& data) {
    // The fit method applies the PCA algorithm to the input data. 
    // It calculates the mean of each feature, centers the data by subtracting the mean, 
    // computes the covariance matrix of the centered data, 
    // and then derives the principal components by obtaining the eigenvectors corresponding to the largest eigenvalues.
    
    // Mean mean of each column in the data matrix.
    mean = data.colwise().mean();  
    // Centers the data by subtracting the mean.
    Eigen::MatrixXd centered_data = data.rowwise() - mean.transpose(); 
    // Covariance matrix.
    Eigen::MatrixXd covariance_matrix = (centered_data.transpose() * centered_data) / (data.rows() - 1);  
    // Eigenvalues and eigenvectors.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(covariance_matrix);

    components = eigensolver.eigenvectors().rightCols(num_components).rowwise().reverse();  
}

Eigen::MatrixXd PCA::transform(const Eigen::MatrixXd& data) {
    // The transform method projects the input data onto the principal component space. 
    // It first centers the new data by subtracting the mean computed during the fitting process, 
    // and then performs a linear transformation using the principal components to obtain the lower-dimensional representation.

     // Centers the new data using the mean calculated during fitting.
    Eigen::MatrixXd centered_data = data.rowwise() - mean.transpose(); 
    
    return centered_data * components; 
}

Eigen::MatrixXd PCA::inverse_transform(const Eigen::MatrixXd& transformed_data) {
    // The inverse_transform method reconstructs the original data from its lower-dimensional representation. 
    // It achieves this by projecting the transformed data back to the original feature space using the principal components,
    // and then adding the mean to restore the original scale.

    return transformed_data * components.transpose() + mean.transpose().replicate(transformed_data.rows(), 1);  
}

Eigen::MatrixXd PCA::get_components() {
    // The get_components method retrieves the principal components that were calculated during the fitting process.
    // These components represent the directions of maximum variance in the data.
    // Returns the principal components calculated during fitting.
    return components;  
}

Eigen::VectorXd PCA::get_mean() {
    // The get_mean method provides the mean vector of the original data, calculated during the fitting process.
    // This mean vector is essential for centering new data before transformation.
    return mean;  
}
