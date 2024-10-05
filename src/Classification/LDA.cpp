#include "LDA.h"
#include <Eigen/Dense>
#include <iostream>
#include <cassert>

using namespace Eigen;

// Constructor for the LDA class, initializing the number of classes to zero.
lda::lda() noexcept : num_classes_(0) {}

// Computes mean vectors for each class from the provided dataset and corresponding labels.
void lda::compute_mean_vectors(const MatrixXd& x_data, const VectorXi& labels) noexcept {
    // The number of distinct classes is determined based on the maximum value in the label array.
    num_classes_ = labels.maxCoeff() + 1;
    // This will hold mean vectors for each class
    mean_vectors_ = MatrixXd(x_data.cols(), num_classes_);

    // For each class, the mean vector is computed by aggregating the corresponding samples.
    for (int class_index = 0; class_index < num_classes_; ++class_index) {
        // Class-specific data is isolated by subtracting the mean vector for the current class.
        MatrixXd class_data = x_data.array().rowwise() - mean_vectors_.col(class_index).transpose().array();

        // Retain only those samples that belong to the current class by filtering based on labels.
        class_data = class_data.array().rowwise() * (labels.array() == class_index).cast<double>().transpose().array();
        
        // The mean vector for the current class is derived from the class-specific data.
        mean_vectors_.col(class_index) = class_data.colwise().mean();
    }
}

// Computes within-class and between-class scatter matrices based on the input dataset and class labels.
void lda::compute_scatter_matrices(const MatrixXd& x_data, const VectorXi& labels) noexcept {
    int num_samples = x_data.rows(); // Total number of samples in the dataset
    int num_features = x_data.cols(); // Total number of features per sample

    // Scatter matrices are initialized to zero to begin accumulation of class-specific contributions.
    within_class_scatter_matrix_ = MatrixXd::Zero(num_features, num_features);
    between_class_scatter_matrix_ = MatrixXd::Zero(num_features, num_features);

    // The overall mean of the dataset is computed, serving as a reference for between-class scatter.
    VectorXd overall_mean = x_data.colwise().mean();

    // For each class, the contribution to the scatter matrices is computed.
    for (int class_index = 0; class_index < num_classes_; ++class_index) {
        // Class samples are isolated by subtracting the mean vector for that class.
        MatrixXd class_samples = x_data.array().rowwise() - mean_vectors_.col(class_index).transpose().array();
        
        // Within-class scatter is calculated as the squared deviations of the samples from their mean.
        MatrixXd class_scatter = class_samples.array().square().matrix();
        within_class_scatter_matrix_ += (class_scatter.transpose() * (labels.array() == class_index).cast<double>()).matrix();

        // Between-class scatter is derived from the mean difference of the class and the overall mean, weighted by the number of samples in that class.
        VectorXd mean_difference = mean_vectors_.col(class_index) - overall_mean;
        between_class_scatter_matrix_ += (mean_difference * mean_difference.transpose()) * ((labels.array() == class_index).cast<double>().sum());
    }
}

// Computes the projection matrix using Eigenvalue Decomposition of the scatter matrices.
void lda::compute_projection_matrix() noexcept {
    // The generalized eigenvalue problem is solved for the ratio of the between-class scatter to within-class scatter.
    SelfAdjointEigenSolver<MatrixXd> solver(between_class_scatter_matrix_.inverse() * within_class_scatter_matrix_);

    // The projection matrix is formed by extracting the eigenvectors corresponding to the largest eigenvalues.
    projection_matrix_ = solver.eigenvectors().transpose();
}

// Fits the LDA model to the provided dataset and class labels, computing necessary statistics for transformation.
void lda::fit(const MatrixXd& x_data, const VectorXi& labels) noexcept {
    // The input dimensions are validated to ensure compatibility between data and labels.
    assert(x_data.rows() == labels.size() && "Number of samples in x_data must match size of labels");
    
    // Mean vectors for each class are computed from the training data.
    compute_mean_vectors(x_data, labels);

    // The within-class and between-class scatter matrices are calculated from the training data.
    compute_scatter_matrices(x_data, labels);

    // The projection matrix is derived based on the computed scatter matrices.
    compute_projection_matrix();
}

// Transforms the input data into a lower-dimensional space using the learned projection matrix.
MatrixXd lda::transform(const MatrixXd& x_data) const noexcept {
    // Input dimension compatibility is ensured prior to transformation.
    assert(x_data.cols() == mean_vectors_.rows() && "Feature dimension must match the training data");

    // The transformation is executed by applying the projection matrix to the input data.
    return x_data * projection_matrix_.transpose();
}
