#pragma once

#include "QDA.h"

namespace mlpp::classifiers {

template<typename Scalar, typename LabelIndex>
QDA<Scalar, LabelIndex>::QDA() : num_classes_(0) {}


// Compute Means
template<typename Scalar, typename LabelIndex>
void QDA<Scalar, LabelIndex>::compute_class_means(
    const Matrix& X, const Labels& labels)
{
    num_classes_ = labels.maxCoeff() + 1;
    const int n_features = X.cols();

    means_.assign(num_classes_, Vector::Zero(n_features));
    std::vector<Scalar> class_count(num_classes_, Scalar(0));

    for (int i = 0; i < X.rows(); ++i) {
        LabelIndex c = labels(i);
        means_[c] += X.row(i).transpose();
        class_count[c] += Scalar(1);
    }

    for (int c = 0; c < num_classes_; ++c) {
        if (class_count[c] > 0)
            means_[c] /= class_count[c];
    }

    // Compute priors
    class_priors_.resize(num_classes_);
    for (int c = 0; c < num_classes_; ++c)
        class_priors_[c] = class_count[c] / Scalar(X.rows());
}


// Compute Covariances
template<typename Scalar, typename LabelIndex>
void QDA<Scalar, LabelIndex>::compute_class_covariances(
    const Matrix& X, const Labels& labels)
{
    const int n_features = X.cols();

    covariances_.assign(num_classes_, Matrix::Zero(n_features, n_features));
    inv_cov_.assign(num_classes_, Matrix());
    log_det_cov_.assign(num_classes_, Scalar(0));

    std::vector<Scalar> class_count(num_classes_, Scalar(0));

    for (int i = 0; i < X.rows(); ++i) {
        LabelIndex c = labels(i);
        Vector diff = X.row(i).transpose() - means_[c];
        covariances_[c] += diff * diff.transpose();
        class_count[c] += Scalar(1);
    }

    // Normalization and we get inverses & determinants
    for (int c = 0; c < num_classes_; ++c) {
        if (class_count[c] > 1)
            covariances_[c] /= (class_count[c] - 1);

        // To avoid singular matrices
        covariances_[c] += Matrix::Identity(n_features, n_features)
                           * Scalar(1e-6);

        // Inverse
        inv_cov_[c] = covariances_[c].inverse();

        // log determinant
        Eigen::FullPivLU<Matrix> lu(covariances_[c]);
        Scalar det = lu.determinant();
        log_det_cov_[c] = std::log(std::abs(det) + Scalar(1e-12));
    }
}


// Fit
template<typename Scalar, typename LabelIndex>
void QDA<Scalar, LabelIndex>::fit(const Matrix& X, const Labels& labels)
{
    if (X.rows() != labels.size())
        throw std::invalid_argument("X rows must match labels length in QDA::fit.");

    compute_class_means(X, labels);
    compute_class_covariances(X, labels);
}


// Log-likelihood
template<typename Scalar, typename LabelIndex>
typename QDA<Scalar, LabelIndex>::Matrix
QDA<Scalar, LabelIndex>::predict_log_likelihood(const Matrix& X) const
{
    const int n_samples = X.rows();
    const int n_features = X.cols();

    Matrix log_probs(n_samples, num_classes_);

    for (int i = 0; i < n_samples; ++i) {
        Vector x = X.row(i).transpose();

        for (int c = 0; c < num_classes_; ++c) {
            Vector diff = x - means_[c];

            // Quadratic form
            Scalar quad = diff.transpose() * inv_cov_[c] * diff;

            // Log-likelihood for QDA
            log_probs(i, c) =
                -Scalar(0.5) * (quad + log_det_cov_[c])
                + std::log(class_priors_[c] + Scalar(1e-12));
        }
    }

    return log_probs;
}


// Predict Labels
template<typename Scalar, typename LabelIndex>
typename QDA<Scalar, LabelIndex>::Labels
QDA<Scalar, LabelIndex>::predict(const Matrix& X) const
{
    Matrix log_probs = predict_log_likelihood(X);

    Labels pred(X.rows());

    for (int i = 0; i < X.rows(); ++i) {
        Eigen::Index max_idx;
        log_probs.row(i).maxCoeff(&max_idx);
        pred(i) = static_cast<LabelIndex>(max_idx);
    }

    return pred;
}

} // namespace mlpp::classifiers
