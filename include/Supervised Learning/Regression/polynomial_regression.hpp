#pragma once

#include "linear_regression.hpp"

#include <Eigen/Dense>
#include <stdexcept>
#include <vector>
#include <cstddef>

namespace mlpp::regression {

/**
 * Polynomial feature expansion for use with any regression model that accepts
 * an Eigen feature matrix.
 *
 * For a d-dimensional input and degree D, two expansion modes are available:
 *
 *   Pure powers (include_interactions = false):
 *     Appends x₁, x₁², …, x₁ᴰ, x₂, …, xᵈᴰ  —  d·D features (+ 1 bias column).
 *     Total output width: (include_bias ? 1 : 0) + d·D.
 *
 *   Full interaction terms (include_interactions = true):
 *     All monomials x₁^α₁ ··· xᵈ^αᵈ with Σαᵢ ≤ D  (excluding the zero monomial
 *     unless include_bias = true).  Total output width: C(d+D, D) [with bias].
 *     This is the standard polynomial feature set used in kernel approximation.
 *
 * The exponent table is built once at construction and reused across all calls
 * to transform(), so repeated transforms on batches of data are efficient.
 *
 * @tparam Scalar  Floating-point type (float, double, long double).
 */
template <typename Scalar = double>
class PolynomialFeatures {
public:
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using Index  = Eigen::Index;

    /**
     * @param degree               Maximum monomial degree D >= 1.
     * @param include_bias         Whether to prepend a constant-1 column
     * @param include_interactions Whether to include cross-feature monomials 
     */
    explicit PolynomialFeatures(unsigned degree               = 2,
                                bool     include_bias         = true,
                                bool     include_interactions = false);

    /**
     * @brief Expand X into polynomial feature space.
     * @param X  Input matrix, shape (n_samples, n_features).
     * @return   Expanded matrix, shape (n_samples, output_dim(n_features)).
     */
    [[nodiscard]] Matrix transform(const Matrix& X) const;

    /// Number of output columns for a given input feature count.
    [[nodiscard]] std::size_t output_dim(std::size_t n_features) const;

    unsigned degree()               const noexcept { return degree_; }
    bool     include_bias()         const noexcept { return include_bias_; }
    bool     include_interactions() const noexcept { return include_interactions_; }

private:
    unsigned degree_;
    bool     include_bias_;
    bool     include_interactions_;

    /// Precomputed exponent table: each row is a multi-index α ∈ ℕ₀ᵈ.
    /// Built lazily on the first call to transform() for a given input width.
    mutable std::vector<std::vector<unsigned>> exponent_table_;
    mutable std::size_t                        table_n_features_ = 0;

    void build_exponent_table(std::size_t n_features) const;

    void recurse_interactions(std::size_t              pos,
                              std::size_t              n_features,
                              unsigned                 remaining,
                              std::vector<unsigned>&   current,
                              std::vector<std::vector<unsigned>>& out) const;
};

/**
 * Polynomial Regression: composes PolynomialFeatures with LinearRegression.
 *
 * The pipeline is:
 *
 *   fit:     X  ->  PolynomialFeatures::transform  ->  X̃  ->  LinearRegression::fit
 *   predict: X  ->  PolynomialFeatures::transform  ->  X̃  ->  LinearRegression::predict
 *
 * LinearRegression internally standardises X̃, so the polynomial features do
 * not need to be pre-scaled. The resulting model fits:
 *
 *   ŷ = Σ_{|α|≤D} w_α ∏ xⱼ^αⱼ  +  b
 *
 * All solver options (Cholesky, SVD, JacobiSVD) and L2 regularisation from
 * LinearRegression are available and forwarded at construction time.
 *
 * @tparam Scalar  Floating-point type (float, double, long double).
 */
template <typename Scalar = double>
class PolynomialRegression {
public:
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    /**
     * @param degree               Maximum polynomial degree D ≥ 1.
     * @param include_interactions Whether to include cross-feature monomials.
     * @param fit_intercept        Forwarded to LinearRegression (default: true).
     * @param regularization       L2 penalty λ ≥ 0 forwarded to LinearRegression.
     * @param method               Solve strategy forwarded to LinearRegression.
     */
    explicit PolynomialRegression(
        unsigned                             degree               = 2,
        bool                                 include_interactions = false,
        bool                                 fit_intercept        = true,
        Scalar                               regularization       = Scalar(0),
        LinearRegression<Scalar>::SolveMethod method              =
            LinearRegression<Scalar>::SolveMethod::Auto);

    /**
     * @brief Expand X to degree D and fit the underlying linear model.
     * @param X  Feature matrix, shape (n_samples, n_features).
     * @param y  Target vector, length n_samples.
     */
    void fit(const Matrix& X, const Vector& y);

    /**
     * @brief Expand X and predict targets.
     * @param X  Feature matrix, shape (n_samples, n_features).
     * @return   Predicted values, length n_samples.
     */
    [[nodiscard]] Vector predict(const Matrix& X) const;

    /// R² on the provided data.
    [[nodiscard]] Scalar score(const Matrix& X, const Vector& y) const;

    /// Residual vector  e = y - ŷ.
    [[nodiscard]] Vector residuals(const Matrix& X, const Vector& y) const;

    /// Coefficients of the underlying linear model in the expanded feature space.
    [[nodiscard]] const Vector& coefficients() const;

    /// Intercept of the underlying linear model.
    [[nodiscard]] Scalar intercept() const;

    [[nodiscard]] bool is_fitted() const noexcept { return regressor_.is_fitted(); }
    [[nodiscard]] const PolynomialFeatures<Scalar>& features() const noexcept { return features_; }
    [[nodiscard]] const LinearRegression<Scalar>& regressor() const noexcept { return regressor_; }

private:
    PolynomialFeatures<Scalar> features_;
    LinearRegression<Scalar>   regressor_;
};

} // namespace mlpp::regression

#include "polynomial_regression.inl"