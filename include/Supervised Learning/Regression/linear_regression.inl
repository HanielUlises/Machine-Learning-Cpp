#pragma once

#include "linear_regression.hpp"

#include <Eigen/Dense>
#include <Eigen/SVD>

#include <stdexcept>
#include <cmath>

namespace mlpp::regression {

template <typename Scalar>
LinearRegression<Scalar>::LinearRegression(bool fit_intercept, Scalar regularization, SolveMethod method)
    : fit_intercept_(fit_intercept)
    , lambda_(regularization)
    , method_(method)
{
    if (lambda_ < Scalar(0))
        throw std::invalid_argument("Regularization parameter λ must be >= 0.");
}

template <typename Scalar>
void LinearRegression<Scalar>::fit(const Matrix& X, const Vector& y)
{
    const Index n = X.rows();
    const Index d = X.cols();

    if (n == 0 || d == 0)
        throw std::invalid_argument("fit(): X must be non-empty.");
    if (y.size() != n)
        throw std::invalid_argument("fit(): X and y must have the same number of rows.");

    feature_mean_ = X.colwise().mean();
    Matrix Xc     = X.rowwise() - feature_mean_.transpose();

    // Population std (not Bessel-corrected); constant columns are given σ = 1
    // so the standardised column is identically zero
    feature_std_ = (Xc.array().square().colwise().sum() / Scalar(n)).sqrt().matrix();
    for (Index j = 0; j < d; ++j)
        if (feature_std_(j) == Scalar(0))
            feature_std_(j) = Scalar(1);

    // Centring y decouples the intercept from the regularised solve the bias
    // is recovered analytically and is never penalised.
    target_mean_ = fit_intercept_ ? y.mean() : Scalar(0);
    Vector ys    = y.array() - target_mean_;

    Matrix Xs = Xc.array().rowwise() / feature_std_.transpose().array();

    // Cholesky requires λ > 0 for guaranteed positive-definiteness of XᵀX + nλI;
    // fall back to SVD for pure OLS or under-determined systems.
    SolveMethod effective = method_;
    if (effective == SolveMethod::Auto)
        effective = (n >= d && lambda_ > Scalar(0)) ? SolveMethod::Cholesky : SolveMethod::SVD;

    Vector w_scaled;
    switch (effective) {
        case SolveMethod::Cholesky:  w_scaled = solve_cholesky(Xs, ys); break;
        case SolveMethod::JacobiSVD: w_scaled = solve_jacobi(Xs, ys);   break;
        default:                     w_scaled = solve_svd(Xs, ys);      break;
    }

    unstandardise(w_scaled, target_mean_);
    fitted_ = true;
}

template <typename Scalar>
typename LinearRegression<Scalar>::Vector
LinearRegression<Scalar>::predict(const Matrix& X) const
{
    if (!fitted_)
        throw std::runtime_error("predict(): model has not been fitted.");
    if (X.cols() != coef_.size())
        throw std::invalid_argument("predict(): feature dimension mismatch.");

    return (X * coef_).array() + intercept_;
}

template <typename Scalar>
Scalar LinearRegression<Scalar>::score(const Matrix& X, const Vector& y) const
{
    if (X.rows() != y.size())
        throw std::invalid_argument("score(): X and y must have the same number of rows.");

    const Vector y_hat  = predict(X);
    const Scalar y_mean = y.mean();
    const Scalar ss_res = (y - y_hat).squaredNorm();
    const Scalar ss_tot = (y.array() - y_mean).matrix().squaredNorm();

    if (ss_tot == Scalar(0)) return Scalar(0);
    return Scalar(1) - ss_res / ss_tot;
}

template <typename Scalar>
typename LinearRegression<Scalar>::Vector
LinearRegression<Scalar>::residuals(const Matrix& X, const Vector& y) const
{
    return y - predict(X);
}

template <typename Scalar>
typename LinearRegression<Scalar>::Vector
LinearRegression<Scalar>::gradient(const Matrix& X, const Vector& y) const
{
    if (!fitted_)
        throw std::runtime_error("gradient(): model has not been fitted.");

    const Index  n = X.rows();
    Vector       g = (X.transpose() * (predict(X) - y)) / Scalar(n);

    if (lambda_ > Scalar(0))
        g += lambda_ * coef_;

    return g;
}

template <typename Scalar>
const typename LinearRegression<Scalar>::Vector&
LinearRegression<Scalar>::coefficients() const
{
    if (!fitted_) throw std::runtime_error("coefficients(): model has not been fitted.");
    return coef_;
}

template <typename Scalar>
Scalar LinearRegression<Scalar>::intercept() const
{
    if (!fitted_) throw std::runtime_error("intercept(): model has not been fitted.");
    return intercept_;
}

template <typename Scalar>
Scalar LinearRegression<Scalar>::condition_number() const
{
    if (cond_number_ < Scalar(0))
        throw std::runtime_error("condition_number(): only available after an SVD solve.");
    return cond_number_;
}

template <typename Scalar>
typename LinearRegression<Scalar>::Matrix
LinearRegression<Scalar>::standardise(const Matrix& X) const
{
    return (X.rowwise() - feature_mean_.transpose()).array()
               .rowwise() / feature_std_.transpose().array();
}

template <typename Scalar>
typename LinearRegression<Scalar>::Vector
LinearRegression<Scalar>::solve_cholesky(const Matrix& Xs, const Vector& ys) const
{
    const Index d = Xs.cols();
    const Index n = Xs.rows();

    Matrix A = Xs.transpose() * Xs;
    // Scale λ by n so the regularisation strength is invariant to sample count,
    // matching the (1/2n) normalisation of the loss. Without this, doubling the
    // dataset would halve the effective penalty.
    if (lambda_ > Scalar(0))
        A += (Scalar(n) * lambda_) * Matrix::Identity(d, d);

    const Eigen::LDLT<Matrix> ldlt(A);
    if (ldlt.info() != Eigen::Success)
        throw std::runtime_error(
            "solve_cholesky(): LDLT factorisation failed. "
            "Try SolveMethod::SVD or increase regularization.");

    return ldlt.solve(Xs.transpose() * ys);
}

template <typename Scalar>
typename LinearRegression<Scalar>::Vector
LinearRegression<Scalar>::solve_svd(const Matrix& Xs, const Vector& ys)
{
    const Index n = Xs.rows();

    Eigen::BDCSVD<Matrix> svd(Xs, Eigen::ComputeThinU | Eigen::ComputeThinV);

    const Vector& sigma = svd.singularValues();
    const Index   r     = sigma.size();
    const Scalar  reg   = Scalar(n) * lambda_;

    Vector filter(r);
    for (Index i = 0; i < r; ++i) {
        const Scalar s = sigma(i);
        // Tikhonov filter: shrinks small singular values rather than truncating them,
        // giving a continuous trade-off between variance reduction and bias.
        filter(i) = s / (s * s + reg);
    }

    // Condition number is meaningful only when all singular values are positive;
    // a zero σ_min indicates rank deficiency and yields κ = ∞.
    if (sigma(0) > Scalar(0)) {
        const Scalar smin = sigma(r - 1);
        cond_number_ = smin > Scalar(0)
                       ? sigma(0) / smin
                       : std::numeric_limits<Scalar>::infinity();
    }

    return svd.matrixV() * (filter.asDiagonal() * (svd.matrixU().transpose() * ys));
}

template <typename Scalar>
typename LinearRegression<Scalar>::Vector
LinearRegression<Scalar>::solve_jacobi(const Matrix& Xs, const Vector& ys)
{
    const Index n = Xs.rows();

    Eigen::JacobiSVD<Matrix> svd(Xs, Eigen::ComputeThinU | Eigen::ComputeThinV);

    const Vector& sigma = svd.singularValues();
    const Index   r     = sigma.size();
    const Scalar  reg   = Scalar(n) * lambda_;

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

    return svd.matrixV() * (filter.asDiagonal() * (svd.matrixU().transpose() * ys));
}

template <typename Scalar>
void LinearRegression<Scalar>::unstandardise(const Vector& w_scaled, Scalar y_mean)
{
    // ŷ = Σ w̃ⱼ (xⱼ−μⱼ)/σⱼ + ȳ  ≡  Σ (w̃ⱼ/σⱼ) xⱼ  +  (ȳ − μᵀw)
    coef_      = w_scaled.array() / feature_std_.array();
    intercept_ = fit_intercept_ ? y_mean - feature_mean_.dot(coef_) : Scalar(0);
}

} // namespace mlpp::regression