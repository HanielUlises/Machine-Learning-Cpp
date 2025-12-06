#pragma once

#include "LDA.h"

template<typename Scalar, typename LabelIndex>
LDA<Scalar, LabelIndex>::LDA()
    : num_classes_(0)
{}

// Compute mean vectors for each class
template<typename Scalar, typename LabelIndex>
void LDA<Scalar, LabelIndex>::compute_mean_vectors(const Matrix& X,
                                                   const Labels& labels)
{
    num_classes_ = labels.maxCoeff() + 1;
    const int n_features = X.cols();

    mean_vectors_.setZero(n_features, num_classes_);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> class_count(num_classes_);
    class_count.setZero();

    // Accumulate sums per class
    for (int i = 0; i < X.rows(); ++i) {
        LabelIndex c = labels(i);
        mean_vectors_.col(c) += X.row(i).transpose();
        class_count(c) += Scalar(1);
    }

    // Divide to get means
    for (int c = 0; c < num_classes_; ++c) {
        if (class_count(c) > Scalar(0))
            mean_vectors_.col(c) /= class_count(c);
    }
}

// Compute scatter matrices Sw and Sb
template<typename Scalar, typename LabelIndex>
void LDA<Scalar, LabelIndex>::compute_scatter_matrices(const Matrix& X,
                                                       const Labels& labels)
{
    const int n_features = X.cols();

    within_class_scatter_matrix_.setZero(n_features, n_features);
    between_class_scatter_matrix_.setZero(n_features, n_features);

    // Compute global mean
    Vector overall_mean = X.colwise().mean();

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> class_count(num_classes_);
    class_count.setZero();

    // Count samples per class
    for (int i = 0; i < labels.size(); ++i)
        class_count(labels(i)) += Scalar(1);

    // Accumulate scatter contributions
    for (int c = 0; c < num_classes_; ++c) {
        if (class_count(c) == Scalar(0))
            continue;

        // Within-class scatter
        for (int i = 0; i < X.rows(); ++i) {
            if (labels(i) == c) {
                Vector diff = X.row(i).transpose() - mean_vectors_.col(c);
                within_class_scatter_matrix_ += diff * diff.transpose();
            }
        }

        // Between-class scatter
        Vector mean_diff = mean_vectors_.col(c) - overall_mean;
        between_class_scatter_matrix_ +=
            class_count(c) * (mean_diff * mean_diff.transpose());
    }
}

// Compute projection matrix
template<typename Scalar, typename LabelIndex>
void LDA<Scalar, LabelIndex>::compute_projection_matrix(int num_components)
{
    // Handle default component count
    if (num_components <= 0)
        num_components = num_classes_ - 1;

    // Regularize Sw slightly to prevent singularity
    Matrix Sw_reg = within_class_scatter_matrix_;
    Sw_reg += Matrix::Identity(Sw_reg.rows(), Sw_reg.cols()) * Scalar(1e-6);

    // Solve the generalized eigenvalue problem: Sb v = λ Sw v
    Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> solver(
        between_class_scatter_matrix_, Sw_reg
    );

    if (solver.info() != Eigen::Success)
        throw std::runtime_error("Eigen decomposition failed in LDA");

    // Select top eigenvectors (eigenvalues sorted ascending)
    const Matrix& eigenvectors = solver.eigenvectors();
    projection_matrix_ = eigenvectors.rightCols(num_components);
}

// Fit LDA model
template<typename Scalar, typename LabelIndex>
void LDA<Scalar, LabelIndex>::fit(const Matrix& X,
                                  const Labels& labels,
                                  int num_components)
{
    if (X.rows() != labels.size())
        throw std::invalid_argument("X rows must match label size");

    compute_mean_vectors(X, labels);
    compute_scatter_matrices(X, labels);
    compute_projection_matrix(num_components);
}

// Transform new data
template<typename Scalar, typename LabelIndex>
typename LDA<Scalar, LabelIndex>::Matrix
LDA<Scalar, LabelIndex>::transform(const Matrix& X) const
{
    if (X.cols() != projection_matrix_.rows())
        throw std::invalid_argument("Feature dimension mismatch in LDA transform");

    return X * projection_matrix_;
}
