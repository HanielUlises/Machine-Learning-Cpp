#pragma once

#include "multilinear_regression.hpp"

#include <Eigen/Dense>
#include <Eigen/SVD>

#include <stdexcept>
#include <cmath>
#include <limits>

namespace mlpp::regression {

template <typename Scalar>
MultilinearRegression<Scalar>::MultilinearRegression(bool fit_intercept, Scalar regularization, SolveMethod method)
    : fit_intercept_(fit_intercept)
    , lambda_(regularization)
    , method_(method)
{
    if (lambda_ < Scalar(0))
        throw std::invalid_argument("Regularization parameter λ must be >= 0.");
}

template <typename Scalar>
void MultilinearRegression<Scalar>::fit(const Matrix& X, const Matrix& Y)
{
    const Index n = X.rows();
    const Index d = X.cols();
    const Index k = Y.cols();

    if (n == 0 || d == 0)
        throw std::invalid_argument("fit(): X must be non-empty.");
    if (Y.rows() != n)
        throw std::invalid_argument("fit(): X and Y must have the same number of rows.");
    if (k == 0)
        throw std::invalid_argument("fit(): Y must have at least one response column.");

    feature_mean_ = X.colwise().mean();
    Matrix Xc     = X.rowwise() - feature_mean_.transpose();

    // Population std; constant columns receive σ = 1 to avoid division by zero
    // while leaving the corresponding standardised column as all-zeros.
    feature_std_ = (Xc.array().square().colwise().sum() / Scalar(n)).sqrt().matrix();
    for (Index j = 0; j < d; ++j)
        if (feature_std_(j) == Scalar(0))
            feature_std_(j) = Scalar(1);

    // Each response is centred independently so the k bias terms are decoupled
    // from the regularised solve and recovered without penalty in unstandardise().
    if (fit_intercept_) {
        target_mean_ = Y.colwise().mean();
    } else {
        target_mean_ = Vector::Zero(k);
    }

    Matrix Ys = Y.rowwise() - target_mean_.transpose();
    Matrix Xs = Xc.array().rowwise() / feature_std_.transpose().array();

    // Cholesky requires λ > 0 for guaranteed positive-definiteness of XᵀX + nλI;
    // fall back to SVD for pure OLS or under-determined systems.
    SolveMethod effective = method_;
    if (effective == SolveMethod::Auto)
        effective = (n >= d && lambda_ > Scalar(0)) ? SolveMethod::Cholesky : SolveMethod::SVD;

    Matrix W_scaled;
    switch (effective) {
        case SolveMethod::Cholesky:  W_scaled = solve_cholesky(Xs, Ys); break;
        case SolveMethod::JacobiSVD: W_scaled = solve_jacobi(Xs, Ys);   break;
        default:                     W_scaled = solve_svd(Xs, Ys);      break;
    }

    unstandardise(W_scaled, target_mean_);
    fitted_ = true;
}

template <typename Scalar>
typename MultilinearRegression<Scalar>::Matrix
MultilinearRegression<Scalar>::predict(const Matrix& X) const
{
    if (!fitted_)
        throw std::runtime_error("predict(): model has not been fitted.");
    if (X.cols() != coef_.rows())
        throw std::invalid_argument("predict(): feature dimension mismatch.");

    return (X * coef_).rowwise() + intercepts_.transpose();
}

template <typename Scalar>
Scalar MultilinearRegression<Scalar>::score(const Matrix& X, const Matrix& Y) const
{
    return score_per_response(X, Y).mean();
}

template <typename Scalar>
typename MultilinearRegression<Scalar>::Vector
MultilinearRegression<Scalar>::score_per_response(const Matrix& X, const Matrix& Y) const
{
    if (X.rows() != Y.rows())
        throw std::invalid_argument("score_per_response(): X and Y row count mismatch.");

    const Matrix Y_hat = predict(X);
    const Index  k     = Y.cols();
    Vector       r2(k);

    for (Index c = 0; c < k; ++c) {
        const Scalar y_mean = Y.col(c).mean();
        const Scalar ss_res = (Y.col(c) - Y_hat.col(c)).squaredNorm();
        const Scalar ss_tot = (Y.col(c).array() - y_mean).matrix().squaredNorm();
        r2(c) = ss_tot == Scalar(0) ? Scalar(0) : Scalar(1) - ss_res / ss_tot;
    }

    return r2;
}

template <typename Scalar>
typename MultilinearRegression<Scalar>::Matrix
MultilinearRegression<Scalar>::residuals(const Matrix& X, const Matrix& Y) const
{
    return Y - predict(X);
}

template <typename Scalar>
typename MultilinearRegression<Scalar>::Matrix
MultilinearRegression<Scalar>::gradient(const Matrix& X, const Matrix& Y) const
{
    if (!fitted_)
        throw std::runtime_error("gradient(): model has not been fitted.");

    const Index n = X.rows();
    Matrix      G = (X.transpose() * (predict(X) - Y)) / Scalar(n);

    if (lambda_ > Scalar(0))
        G += lambda_ * coef_;

    return G;
}

template <typename Scalar>
const typename MultilinearRegression<Scalar>::Matrix&
MultilinearRegression<Scalar>::coefficients() const
{
    if (!fitted_) throw std::runtime_error("coefficients(): model has not been fitted.");
    return coef_;
}

template <typename Scalar>
const typename MultilinearRegression<Scalar>::Vector&
MultilinearRegression<Scalar>::intercepts() const
{
    if (!fitted_) throw std::runtime_error("intercepts(): model has not been fitted.");
    return intercepts_;
}

template <typename Scalar>
Scalar MultilinearRegression<Scalar>::condition_number() const
{
    if (cond_number_ < Scalar(0))
        throw std::runtime_error("condition_number(): only available after an SVD solve.");
    return cond_number_;
}

template <typename Scalar>
typename MultilinearRegression<Scalar>::Matrix
MultilinearRegression<Scalar>::standardise(const Matrix& X) const
{
    return (X.rowwise() - feature_mean_.transpose()).array()
               .rowwise() / feature_std_.transpose().array();
}

template <typename Scalar>
typename MultilinearRegression<Scalar>::Matrix
MultilinearRegression<Scalar>::solve_cholesky(const Matrix& Xs, const Matrix& Ys) const
{
    const Index d = Xs.cols();
    const Index n = Xs.rows();

    Matrix A = Xs.transpose() * Xs;
    // λ is scaled by n for sample-count invariance (see LinearRegression::solve_cholesky).
    // A single LDLT factorisation amortises the O(d³) cost across all k right-hand sides.
    if (lambda_ > Scalar(0))
        A += (Scalar(n) * lambda_) * Matrix::Identity(d, d);

    const Eigen::LDLT<Matrix> ldlt(A);
    if (ldlt.info() != Eigen::Success)
        throw std::runtime_error(
            "solve_cholesky(): LDLT factorisation failed. "
            "Try SolveMethod::SVD or increase regularization.");

    // ldlt.solve() accepts a matrix RHS, solving all k columns in one triangular sweep.
    return ldlt.solve(Xs.transpose() * Ys);
}

template <typename Scalar>
typename MultilinearRegression<Scalar>::Matrix
MultilinearRegression<Scalar>::solve_svd(const Matrix& Xs, const Matrix& Ys)
{
    const Index n   = Xs.rows();
    const Scalar reg = Scalar(n) * lambda_;

    Eigen::BDCSVD<Matrix> svd(Xs, Eigen::ComputeThinU | Eigen::ComputeThinV);

    const Vector& sigma = svd.singularValues();
    const Index   r     = sigma.size();

    Vector filter(r);
    for (Index i = 0; i < r; ++i) {
        const Scalar s = sigma(i);
        // Tikhonov filter shared across all k responses; the decomposition is
        // computed once and applied via a single matrix multiply UᵀY ∈ ℝʳˣᵏ.
        filter(i) = s / (s * s + reg);
    }

    if (sigma(0) > Scalar(0)) {
        const Scalar smin = sigma(r - 1);
        cond_number_ = smin > Scalar(0)
                       ? sigma(0) / smin
                       : std::numeric_limits<Scalar>::infinity();
    }

    return svd.matrixV() * (filter.asDiagonal() * (svd.matrixU().transpose() * Ys));
}

template <typename Scalar>
typename MultilinearRegression<Scalar>::Matrix
MultilinearRegression<Scalar>::solve_jacobi(const Matrix& Xs, const Matrix& Ys)
{
    const Index n   = Xs.rows();
    const Scalar reg = Scalar(n) * lambda_;

    Eigen::JacobiSVD<Matrix> svd(Xs, Eigen::ComputeThinU | Eigen::ComputeThinV);

    const Vector& sigma = svd.singularValues();
    const Index   r     = sigma.size();

    Vector filter(r);
    for (Index i = 0; i < r; ++i) {
        const Scalar s = sigma(i);
        filter(i) = s / (s * s + reg);
    }

    if (sigma(0) > Scalar(0)) {
        const Scalar smin = sigma(r - 1);
        cond_number_ = smin > Scalar(0)
                       ? sigma(0) / smin
                       : std::numeric_limits<Scalar>::infinity();
    }

    return svd.matrixV() * (filter.asDiagonal() * (svd.matrixU().transpose() * Ys));
}

template <typename Scalar>
void MultilinearRegression<Scalar>::unstandardise(const Matrix& W_scaled, const Vector& y_mean)
{
    // W_orig_jk = W̃_jk / σ_j  —  each row scaled by the reciprocal of its feature std.
    // Expanding ŷₖ = X̃ w̃ₖ + ȳₖ yields this form after collecting terms on xⱼ.
    coef_ = W_scaled.array().colwise() / feature_std_.array();

    // b_k = ȳ_k − μᵀ wₖ  —  one intercept per response, none regularised.
    if (fit_intercept_) {
        intercepts_ = y_mean - (feature_mean_.transpose() * coef_).transpose();
    } else {
        intercepts_ = Vector::Zero(W_scaled.cols());
    }
}

} // namespace mlpp::regression