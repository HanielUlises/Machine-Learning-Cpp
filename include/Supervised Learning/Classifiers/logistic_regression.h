#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include <Eigen/Dense>
#include <vector>
#include <cstddef>

namespace mlpp::classifiers{
/**
 * @brief Binary Logistic Regression implemented via gradient descent.
 *
 * This class performs maximum-likelihood estimation of the parameter vector θ ∈ ℝ^{d+1}
 * (including intercept) for binary classification under the logistic (sigmoid) model:
 *
 *     P(y = 1 | x; θ) = σ(θᵀ x̃) ,   where σ(z) = 1/(1 + exp(-z))
 *     x̃ = [1, x] is the feature vector augmented with a constant 1 for the bias term.
 *
 * The objective minimized is the average negative log-likelihood (binary cross-entropy):
 *
 *     L(θ) = - (1/n) Σ [ yᵢ log pᵢ + (1-yᵢ) log (1-pᵢ) ]
 *
 * Training is performed using vanilla gradient descent with a fixed learning rate.
 * Convergence is declared when the infinity-norm of the parameter update falls below tol.
 */

template<typename Scalar = double>
class LogisticRegressionBinary {
public:
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Index = Eigen::Index;

    LogisticRegressionBinary() = default;

    // Fit binary logistic regression (labels must be 0 or 1)
    void fit(const Matrix& X, const Vector& y,
             Scalar learning_rate = Scalar(0.01),
             std::size_t max_iter = 1000,
             Scalar tol = Scalar(1e-6));

    // Predict probabilities for binary case
    Vector predict_proba(const Matrix& X) const;

    // Predict class labels (threshold = 0.5)
    Vector predict(const Matrix& X, Scalar threshold = Scalar(0.5)) const;

    // Getter
    const Vector& coefficients() const { return theta_; }
    Scalar intercept() const { return theta_(0); }

private:
    Vector theta_;  // first element is intercept
    static Scalar sigmoid(Scalar z) { return Scalar(1) / (Scalar(1) + std::exp(-z)); }
};

/**
 * @brief Multiclass Logistic Regression (softmax regression) via One-vs-Rest training.
 *
 * This class implements multinomial logistic regression for K ≥ 2 classes.
 * Internally, K independent binary logistic models are trained in a one-versus-rest fashion.
 * Each row k of the coefficient matrix Θ ∈ ℝ^{K × (d+1)} corresponds to the parameter vector
 * for class k (including intercept term).
 *
 * Raw sigmoid outputs from the K binary models are normalized row-wise to produce valid
 * class probabilities:
 *
 *     P(y = k | x) = σ(Θ_k x̃) / Σ_{j=1}^K σ(Θ_j x̃)
 *
 * This normalization yields an approximation of the true softmax regression probabilities
 * while retaining the simplicity of independent binary learners.
 *
 * Prediction returns the class index with the highest estimated probability.
 */

template<typename Scalar = double>
class LogisticRegressionMulti {
public:
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Index = Eigen::Index;

    LogisticRegressionMulti() = default;

    // Fit multiclass logistic regression (one-vs-rest by default)
    // labels y must be integer class indices starting from 0
    void fit(const Matrix& X, const Vector& y,
             Scalar learning_rate = Scalar(0.01),
             std::size_t max_iter = 1000,
             Scalar tol = Scalar(1e-6));

    // Predict class probabilities: rows = samples, cols = classes
    Matrix predict_proba(const Matrix& X) const;

    // Predict class labels
    Vector predict(const Matrix& X) const;

    // Coefficients: rows = classes, cols = features+1 (including intercept)
    const Matrix& coefficients() const { return thetas_; }

private:
    Matrix thetas_;  // each row is a binary classifier (intercept in first column)
    int n_classes_ = 0;
    static Scalar sigmoid(Scalar z) { return Scalar(1) / (Scalar(1) + std::exp(-z)); }
};

#include "logistic_regression.inl"

#endif // LOGISTIC_REGRESSION_H

}