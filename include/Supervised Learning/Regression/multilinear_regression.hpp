#pragma once

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <stdexcept>
#include <cstddef>

namespace mlpp::regression {

/**
 * Multilinear (multivariate) Regression — k response variables fitted simultaneously.
 *
 * Solves:  min_W  (1/2n) ||XW - Y||²_F  +  (λ/2) ||W||²_F
 *
 * where  X ∈ ℝⁿˣᵈ,  Y ∈ ℝⁿˣᵏ,  W ∈ ℝᵈˣᵏ.
 *
 * The Frobenius-norm objective decouples column-wise, so each response yₖ is
 * mathematically independent. The key HPC advantage over running k separate
 * LinearRegression instances is that the coefficient matrix A = XᵀX + nλI is
 * factored exactly once, and all k right-hand sides are solved in a single call:
 *
 *   Cholesky  —  LDLT(A).solve(XᵀY)              O(nd² + d²k)
 *   SVD       —  V diag(σᵢ/(σᵢ²+nλ)) UᵀY        one decomposition, k solves
 *   JacobiSVD —  same filter, full-pivoting SVD
 *
 * Standardisation is per-feature (shared μ, σ across all responses). Each
 * response column is mean-centred independently when fit_intercept = true,
 * giving a bias vector b ∈ ℝᵏ rather than a scalar.
 *
 * @tparam Scalar  Floating-point type (float, double, long double).
 */
template <typename Scalar = double>
class MultilinearRegression {
public:
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Index  = Eigen::Index;

    enum class SolveMethod {
        Auto,       ///< Chosen automatically (recommended)
        Cholesky,   ///< Normal equations via LDLT; O(nd² + d²k), fast for n >> d
        SVD,        ///< Thin BDCSVD; O(nd·min(n,d)), stable for any shape
        JacobiSVD   ///< Full-pivoting JacobiSVD; slowest but maximally stable
    };

    /**
     * @param fit_intercept   Whether to fit a bias vector b ∈ ℝᵏ (default: true).
     * @param regularization  L2 penalty λ ≥ 0 applied uniformly to all responses (default: 0).
     * @param method          Linear-solve strategy (default: Auto).
     */
    explicit MultilinearRegression(bool        fit_intercept = true,
                                   Scalar      regularization = Scalar(0),
                                   SolveMethod method         = SolveMethod::Auto);

    /**
     * @brief Fit the model to multiple response variables.
     * @param X  Feature matrix, shape (n_samples, n_features).
     * @param Y  Response matrix, shape (n_samples, n_responses).
     */
    void fit(const Matrix& X, const Matrix& Y);

    /**
     * @brief Predict response matrix for new samples.
     * @param X  Feature matrix, shape (n_samples, n_features).
     * @return   Predicted matrix, shape (n_samples, n_responses).
     */
    [[nodiscard]] Matrix predict(const Matrix& X) const;

    /**
     * @brief Mean R² across all response variables.
     * @return Scalar R² ∈ (-∞, 1].
     */
    [[nodiscard]] Scalar score(const Matrix& X, const Matrix& Y) const;

    /**
     * @brief Per-response R² scores.
     * @return Vector of length n_responses; r2(k) = 1 - SS_res_k / SS_tot_k.
     */
    [[nodiscard]] Vector score_per_response(const Matrix& X, const Matrix& Y) const;

    /// Residual matrix  E = Y - XW - 1bᵀ.  Requires fit().
    [[nodiscard]] Matrix residuals(const Matrix& X, const Matrix& Y) const;

    /// Gradient of the regularised MSE:  ∇L = (1/n) Xᵀ(XW − Y) + λW.  Requires fit().
    [[nodiscard]] Matrix gradient(const Matrix& X, const Matrix& Y) const;

    /// Coefficient matrix W ∈ ℝᵈˣᵏ in original (unscaled) feature space.
    [[nodiscard]] const Matrix& coefficients() const;

    /// Bias vector b ∈ ℝᵏ; zero vector if fit_intercept == false.
    [[nodiscard]] const Vector& intercepts() const;

    /// True after a successful call to fit().
    [[nodiscard]] bool is_fitted() const noexcept { return fitted_; }

    /// Condition number σ_max/σ_min of the design matrix. Only available after an SVD solve.
    [[nodiscard]] Scalar condition_number() const;

private:
    bool        fit_intercept_;
    Scalar      lambda_;
    SolveMethod method_;

    Matrix coef_;       ///< W ∈ ℝᵈˣᵏ in original feature space
    Vector intercepts_; ///< b ∈ ℝᵏ

    Vector feature_mean_;   ///< per-feature mean, shape (d,)
    Vector feature_std_;    ///< per-feature std,  shape (d,)
    Vector target_mean_;    ///< per-response mean, shape (k,)

    bool   fitted_      = false;
    Scalar cond_number_ = Scalar(-1);

    [[nodiscard]] Matrix standardise(const Matrix& X) const;

    /// Solve via LDLT of XᵀX + nλI. Returns W in standardised space.
    [[nodiscard]] Matrix solve_cholesky(const Matrix& Xs, const Matrix& Ys) const;

    /// Solve via thin BDCSVD with Tikhonov filter. Returns W in standardised space.
    [[nodiscard]] Matrix solve_svd(const Matrix& Xs, const Matrix& Ys);

    /// Solve via full-pivoting JacobiSVD with Tikhonov filter. Returns W in standardised space.
    [[nodiscard]] Matrix solve_jacobi(const Matrix& Xs, const Matrix& Ys);

    /// Map W from standardised → original space; compute bias vector.
    void unstandardise(const Matrix& W_scaled, const Vector& y_mean);
};

} // namespace mlpp::regression

#include "multilinear_regression.inl"