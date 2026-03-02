#pragma once

#include "polynomial_regression.hpp"

#include <Eigen/Dense>

#include <stdexcept>
#include <cmath>
#include <numeric>

namespace mlpp::regression {

template <typename Scalar>
PolynomialFeatures<Scalar>::PolynomialFeatures(unsigned degree,
                                               bool     include_bias,
                                               bool     include_interactions)
    : degree_(degree < 1 ? 1 : degree)
    , include_bias_(include_bias)
    , include_interactions_(include_interactions)
{}

template <typename Scalar>
typename PolynomialFeatures<Scalar>::Matrix
PolynomialFeatures<Scalar>::transform(const Matrix& X) const
{
    const auto n_features = static_cast<std::size_t>(X.cols());
    const Index n         = X.rows();

    if (n_features == 0)
        throw std::invalid_argument("transform(): X must have at least one feature column.");

    // Rebuild the exponent table only when the input width changes.
    if (n_features != table_n_features_)
        build_exponent_table(n_features);

    const std::size_t out_cols = exponent_table_.size();
    Matrix Xp(n, static_cast<Index>(out_cols));

    for (std::size_t k = 0; k < out_cols; ++k) {
        const auto& alpha = exponent_table_[k];

        // Bias column: all exponents are zero, so the product is identically 1.
        bool is_bias = true;
        for (unsigned e : alpha) if (e != 0) { is_bias = false; break; }
        if (is_bias) {
            Xp.col(static_cast<Index>(k)).setOnes();
            continue;
        }

        // Accumulate the monomial x₁^α₁ ··· xᵈ^αᵈ row-wise
        Xp.col(static_cast<Index>(k)).setOnes();
        for (std::size_t j = 0; j < n_features; ++j) {
            if (alpha[j] == 0) continue;
            for (unsigned p = 0; p < alpha[j]; ++p)
                Xp.col(static_cast<Index>(k)).array() *= X.col(static_cast<Index>(j)).array();
        }
    }

    return Xp;
}

template <typename Scalar>
std::size_t PolynomialFeatures<Scalar>::output_dim(std::size_t n_features) const
{
    if (n_features == 0) return 0;

    if (!include_interactions_) {
        return (include_bias_ ? 1u : 0u) + static_cast<std::size_t>(n_features) * degree_;
    }

    // Number of monomials with total degree ≤ D for F features equals C(F+D, D).
    // Computed with a numerically stable multiplicative formula to avoid overflow
    // for moderate F and D.
    const unsigned F = static_cast<unsigned>(n_features);
    const unsigned D = degree_;
    unsigned k = std::min(D, F);   // choose the smaller argument for C(F+D, D) = C(F+D, F)
    unsigned numer = 1, denom = 1;
    for (unsigned i = 1; i <= k; ++i) {
        numer *= (F + D + 1 - i);
        denom *= i;
    }
    return static_cast<std::size_t>(numer / denom);
}

template <typename Scalar>
void PolynomialFeatures<Scalar>::build_exponent_table(std::size_t n_features) const
{
    exponent_table_.clear();

    if (!include_interactions_) {
        // Pure-power mode: one entry per (feature, degree) pair.
        // The bias column (all zeros) is prepended first if requested.
        if (include_bias_)
            exponent_table_.push_back(std::vector<unsigned>(n_features, 0u));

        for (std::size_t j = 0; j < n_features; ++j)
            for (unsigned p = 1; p <= degree_; ++p) {
                std::vector<unsigned> alpha(n_features, 0u);
                alpha[j] = p;
                exponent_table_.push_back(std::move(alpha));
            }
    } else {
        // Interaction mode: enumerate all multi-indices α with Σαᵢ ≤ D.
        // The zero multi-index (bias) is included iff bias is requested
        if (include_bias_)
            exponent_table_.push_back(std::vector<unsigned>(n_features, 0u));

        // Iterate total degrees 1 … D and collect all compositions at each level.
        for (unsigned total = 1; total <= degree_; ++total) {
            std::vector<unsigned> current(n_features, 0u);
            recurse_interactions(0, n_features, total, current, exponent_table_);
        }
    }

    table_n_features_ = n_features;
}

template <typename Scalar>
void PolynomialFeatures<Scalar>::recurse_interactions(
    std::size_t              pos,
    std::size_t              n_features,
    unsigned                 remaining,
    std::vector<unsigned>&   current,
    std::vector<std::vector<unsigned>>& out) const
{
    if (pos + 1 == n_features) {
        current[pos] = remaining;
        out.push_back(current);
        return;
    }

    for (unsigned v = 0; v <= remaining; ++v) {
        current[pos] = v;
        recurse_interactions(pos + 1, n_features, remaining - v, current, out);
    }
}

template <typename Scalar>
PolynomialRegression<Scalar>::PolynomialRegression(
    unsigned                              degree,
    bool                                  include_interactions,
    bool                                  fit_intercept,
    Scalar                                regularization,
    LinearRegression<Scalar>::SolveMethod method)

    : features_(degree, /*include_bias=*/false, include_interactions)
    , regressor_(fit_intercept, regularization, method)
{}

template <typename Scalar>
void PolynomialRegression<Scalar>::fit(const Matrix& X, const Vector& y)
{
    if (X.rows() == 0)
        throw std::invalid_argument("fit(): X must be non-empty.");
    regressor_.fit(features_.transform(X), y);
}

template <typename Scalar>
typename PolynomialRegression<Scalar>::Vector
PolynomialRegression<Scalar>::predict(const Matrix& X) const
{
    if (!is_fitted())
        throw std::runtime_error("predict(): model has not been fitted.");

    return regressor_.predict(features_.transform(X));
}

template <typename Scalar>
Scalar PolynomialRegression<Scalar>::score(const Matrix& X, const Vector& y) const
{
    if (X.rows() != y.size())
        throw std::invalid_argument("score(): X and y must have the same number of rows.");

    const Matrix Xp    = features_.transform(X);
    const Vector y_hat = regressor_.predict(Xp);
    const Scalar y_mean = y.mean();

    const Scalar ss_res = (y - y_hat).squaredNorm();
    const Scalar ss_tot = (y.array() - y_mean).matrix().squaredNorm();

    if (ss_tot == Scalar(0)) return Scalar(0);
    return Scalar(1) - ss_res / ss_tot;
}

template <typename Scalar>
typename PolynomialRegression<Scalar>::Vector
PolynomialRegression<Scalar>::residuals(const Matrix& X, const Vector& y) const
{
    return y - predict(X);
}

template <typename Scalar>
const typename PolynomialRegression<Scalar>::Vector&
PolynomialRegression<Scalar>::coefficients() const
{
    return regressor_.coefficients();
}

template <typename Scalar>
Scalar PolynomialRegression<Scalar>::intercept() const
{
    return regressor_.intercept();
}

} // namespace mlpp::regression