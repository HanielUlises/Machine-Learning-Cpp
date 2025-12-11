#ifndef LOGISTIC_REGRESSION_INL
#define LOGISTIC_REGRESSION_INL

#include "logistic_regression.h"

#include <Eigen/Dense>
#include <numeric>
#include <cmath>
#include <iostream>

namespace mlpp::classifiers{

template<typename Scalar>
void LogisticRegressionBinary<Scalar>::fit(const Matrix& X, const Vector& y,
                                           Scalar learning_rate,
                                           std::size_t max_iter,
                                           Scalar tol)
{
    Index n_samples = X.rows();
    Index n_features = X.cols();

    // Intercept term
    Matrix X_b = Matrix(n_samples, n_features + 1);
    X_b.leftCols(n_features) = X;
    X_b.col(n_features).setOnes();

    theta_ = Vector::Zero(n_features + 1);

    for (std::size_t iter = 0; iter < max_iter; ++iter) {
        Vector probs = X_b * theta_;
        probs = probs.unaryExpr(&sigmoid);

        Vector error = probs - y;
        Vector grad = (X_b.transpose() * error) / Scalar(n_samples);

        Vector theta_old = theta_;
        theta_ -= learning_rate * grad;

        if ((theta_ - theta_old).array().abs().maxCoeff() < tol) {
            // std::cout << "Binary convergence after " << iter + 1 << " iterations.\n";
            break;
        }
    }
}

template<typename Scalar>
auto LogisticRegressionBinary<Scalar>::predict_proba(const Matrix& X) const -> Vector
{
    Index n_samples = X.rows();
    Index n_features = X.cols();

    Matrix X_b(n_samples, n_features + 1);
    X_b.leftCols(n_features) = X;
    X_b.col(n_features).setOnes();

    return (X_b * theta_).unaryExpr(&sigmoid);
}

template<typename Scalar>
auto LogisticRegressionBinary<Scalar>::predict(const Matrix& X, Scalar threshold) const -> Vector
{
    Vector proba = predict_proba(X);
    return (proba.array() >= threshold).cast<Scalar>();
}

// Multiclass implementation (One-vs-Rest)

template<typename Scalar>
void LogisticRegressionMulti<Scalar>::fit(const Matrix& X, const Vector& y,
                                          Scalar learning_rate,
                                          std::size_t max_iter,
                                          Scalar tol)
{
    Index n_samples = X.rows();
    Index n_features = X.cols();

    // Determine number of unique classes
    std::vector<int> classes;
    for (Index i = 0; i < y.size(); ++i) {
        int c = static_cast<int>(y(i));
        if (std::find(classes.begin(), classes.end(), c) == classes.end())
            classes.push_back(c);
    }
    std::sort(classes.begin(), classes.end());
    n_classes_ = static_cast<int>(classes.size());

    // Map class labels to 0..n_classes_-1 if needed
    Vector y_mapped = y;
    if (classes.front() != 0 || classes.back() != n_classes_ - 1) {
        // create mapping
        y_mapped = Vector::Zero(n_samples);
        for (Index i = 0; i < n_samples; ++i) {
            auto it = std::find(classes.begin(), classes.end(), static_cast<int>(y(i)));
            y_mapped(i) = std::distance(classes.begin(), it);
        }
    }

    // Add intercept term once
    Matrix X_b(n_samples, n_features + 1);
    X_b.leftCols(n_features) = X;
    X_b.col(n_features).setOnes();

    thetas_ = Matrix::Zero(n_classes_, n_features + 1);

    for (int c = 0; c < n_classes_; ++c) {
        Vector y_binary = (y_mapped.array() == c).cast<Scalar>();

        Vector theta = Vector::Zero(n_features + 1);

        for (std::size_t iter = 0; iter < max_iter; ++iter) {
            Vector probs = (X_b * theta).unaryExpr(&sigmoid);
            Vector error = probs - y_binary;
            Vector grad = (X_b.transpose() * error) / Scalar(n_samples);

            Vector theta_old = theta;
            theta -= learning_rate * grad;

            if ((theta - theta_old).array().abs().maxCoeff() < tol)
                break;
        }

        thetas_.row(c) = theta.transpose();
    }
}

template<typename Scalar>
auto LogisticRegressionMulti<Scalar>::predict_proba(const Matrix& X) const -> Matrix
{
    Index n_samples = X.rows();
    Index n_features = X.cols();

    Matrix X_b(n_samples, n_features + 1);
    X_b.leftCols(n_features) = X;
    X_b.col(n_features).setOnes();

    Matrix logits = X_b * thetas_.transpose();  // n_samples x n_classes
    Matrix probs = logits.unaryExpr(&sigmoid);

    // Normalize so probabilities sum to 1 (simple softmax-like normalization for OvR)
    Matrix row_sums = probs.rowwise().sum();
    for (Index i = 0; i < n_samples; ++i) {
        if (row_sums(i) > Scalar(0))
            probs.row(i) /= row_sums(i);
    }

    return probs;
}

template<typename Scalar>
auto LogisticRegressionMulti<Scalar>::predict(const Matrix& X) const -> Vector
{
    Matrix proba = predict_proba(X);
    Vector labels(proba.rows());
    for (Index i = 0; i < proba.rows(); ++i) {
        proba.row(i).maxCoeff(&labels(i));
    }
    return labels;
}

}

#endif // LOGISTIC_REGRESSION_INL