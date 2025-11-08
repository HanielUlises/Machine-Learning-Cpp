#ifndef LDA_H
#define LDA_H

#include <vector>
#include <Eigen/Dense>

// Linear Discriminant Analysis (LDA) class
// Implements LDA for dimensionality reduction and classification.
// The class computes scatter matrices, mean vectors, and projection matrix
// to transform data into a lower-dimensional space maximizing class separability.

class lda {
private:
    // Mean vectors for each class
    Eigen::MatrixXd mean_vectors_;

    // Within-class scatter matrix (Sw)
    Eigen::MatrixXd within_class_scatter_matrix_;

    // Between-class scatter matrix (Sb)
    Eigen::MatrixXd between_class_scatter_matrix_;

    // The projection matrix (transforms data to lower dimension)
    Eigen::MatrixXd projection_matrix_;

    // Number of distinct classes in the dataset
    int num_classes_;

    // Computes mean vectors for each class based on input data (X) and class labels (y)
    void compute_mean_vectors(const Eigen::MatrixXd& x_data, const Eigen::VectorXi& labels) noexcept;

    // Computes the within-class and between-class scatter matrices (Sw and Sb)
    void compute_scatter_matrices(const Eigen::MatrixXd& x_data, const Eigen::VectorXi& labels) noexcept;

    // Computes the projection matrix used to transform the input data
    void compute_projection_matrix() noexcept;

public:
    // Constructor: Initializes the LDA object
    lda() noexcept;

    // Fit function: Learns the LDA model by computing mean vectors, scatter matrices, and projection matrix
    void fit(const Eigen::MatrixXd& x_data, const Eigen::VectorXi& labels) noexcept;

    // Transforms input data (X) into the lower-dimensional LDA space using the computed projection matrix
    Eigen::MatrixXd transform(const Eigen::MatrixXd& x_data) const noexcept;
};

#endif // LDA_H
