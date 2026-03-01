#pragma once

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <stdexcept>
#include <cstddef>

namespace mlpp::regression {

/**
 * @brief Ordinary Least Squares (and optionally L2-regularised) Linear Regression.
 *
 * Solves the problem
 *
 *     min_w  (1/2n) ||Xw - y||² + (λ/2) ||w||²
 *
 * The fit strategy is chosen automatically based on problem geometry:
 *
 *   • n >> d  →  Normal equations via Cholesky (XᵀX + λI) w = Xᵀy
 *                Fast when d is small; O(nd² + d³).
 *
 *   • d >> n  →  Thin SVD of X̃ = UΣVᵀ followed by the closed-form
 *                solution  w = V diag(σᵢ/(σᵢ² + λ)) Uᵀ y
 *                Numerically stable; avoids forming XᵀX.
 *
 *   • ill-conditioned  →  Fall back to JacobiSVD (full pivoting).
 *
 * When fit_intercept = true the data is centred prior to solving
 * (mean-subtraction on X and y), so the bias is never regularised.
 *
 * Feature standardisation (zero-mean, unit-variance) is applied internally
 * before solving; returned coefficients are always in the *original* feature
 * space so callers need not transform their data.
 *
 * @tparam Scalar  Floating-point type (float, double, long double).
 */
template <typename Scalar = double>
class LinearRegression {
public:
    using Matrix   = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using Vector   = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Index    = Eigen::Index;


    enum class SolveMethod {
        Auto,       ///< Chosen automatically (recommended)
        Cholesky,   ///< Normal equations via LDLT; O(nd² + d³), fast for n >> d
        SVD,        ///< Thin BDCSVD; O(nd²) or O(n²d), stable for any shape
        JacobiSVD   ///< Full-pivoting JacobiSVD; slowest but maximally stable
    };

    /**
     * @param fit_intercept       Whether to fit a bias term (default: true).
     * @param regularization      L2 penalty λ ≥ 0 (default: 0 = pure OLS).
     * @param method              Linear-solve strategy (default: Auto).
     */
    explicit LinearRegression(bool        fit_intercept  = true,
                              Scalar      regularization  = Scalar(0),
                              SolveMethod method          = SolveMethod::Auto);

    /**
     * @brief Fit the model.
     *
     * @param X  Feature matrix, shape (n_samples, n_features), row-major.
     * @param y  Target vector, length n_samples.
     */
    void fit(const Matrix& X, const Vector& y);

    /**
     * @brief Predict targets for new samples.
     *
     * @param X  Feature matrix, shape (n_samples, n_features).
     * @return   Predicted values, length n_samples.
     */
    [[nodiscard]] Vector predict(const Matrix& X) const;

    /**
     * @brief Coefficient of determination R².
     *
     *     R² = 1 - SS_res / SS_tot
     *
     * @param X  Feature matrix.
     * @param y  True target values.
     * @return   R² ∈ (-∞, 1].
     */
    [[nodiscard]] Scalar score(const Matrix& X, const Vector& y) const;

    /**
     * @brief Residual vector on training data:  e = y - Xw - b
     *
     * Only valid after fit(); throws otherwise.
     */
    [[nodiscard]] Vector residuals(const Matrix& X, const Vector& y) const;

    /**
     * @brief Gradient of the (regularised) MSE loss w.r.t. w, evaluated at
     *        the current coefficients on the provided data.
     *
     *     ∇L = (1/n) Xᵀ(Xw - y) + λw
     */
    [[nodiscard]] Vector gradient(const Matrix& X, const Vector& y) const;

    /// Coefficient vector in original (unscaled) feature space, length n_features.
    [[nodiscard]] const Vector& coefficients() const;

    /// Intercept (bias) term; 0 if fit_intercept == false.
    [[nodiscard]] Scalar intercept() const;

    /// True after a successful call to fit().
    [[nodiscard]] bool is_fitted() const noexcept { return fitted_; }

    /// Effective condition number of the design matrix (available after SVD solve).
    [[nodiscard]] Scalar condition_number() const;

private:
    //  Hyper-parameters
    bool        fit_intercept_;
    Scalar      lambda_;
    SolveMethod method_;

    //  Learned parameters (original feature space)
    Vector  coef_;           ///< w in original space
    Scalar  intercept_{};    ///< bias

    Vector  feature_mean_;
    Vector  feature_std_;
    Scalar  target_mean_{};

    bool   fitted_       = false;
    Scalar cond_number_  = Scalar(-1);

    [[nodiscard]] Matrix standardise(const Matrix& X) const;

    // Solve the (regularised) normal equations on standardised data.
    // Returns coefficient vector in *scaled* space.
    [[nodiscard]] Vector solve_cholesky(const Matrix& Xs, const Vector& ys) const;
    [[nodiscard]] Vector solve_svd     (const Matrix& Xs, const Vector& ys);
    [[nodiscard]] Vector solve_jacobi  (const Matrix& Xs, const Vector& ys);

    // Convert coefficient vector from standardised → original space. */
    void unstandardise(const Vector& w_scaled, Scalar y_mean);
};

} // namespace mlpp::regression

#include "linear_regression.inl"