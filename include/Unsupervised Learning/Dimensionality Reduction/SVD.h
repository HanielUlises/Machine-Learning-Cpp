#pragma once
#include <Eigen/Dense>
#include <stdexcept>
/*

    Given data matrix X ∈ R^{m×n}, rows as samples, columns as features,
    the thin singular value decomposition is
       X = U Σ V^T,
    where Σ = diag(σ₁,...,σ_r), r = min(m,n), σ₁ ≥ ... ≥ σ_r ≥ 0.

    Dimensionality reduction to k < r is obtained by truncation:
       X_k = U_k Σ_k V_k^T,
    which is the best rank-k approximation to X (Eckart–Young–Mirsky).

    The reduced representation of each sample is given by
       Z = U_k Σ_k  ∈ R^{m×k},
    i.e. coordinates in the principal directions.
*/

class SVDReducer {
public:
    // Fit SVD to data matrix (rows = samples)
    void fit(const Eigen::MatrixXd& X);

    // Reduce to k dimensions, returns Z = U_k Σ_k
    Eigen::MatrixXd transform(std::size_t k) const;

    // Transform a new batch: X_new V_k
    Eigen::MatrixXd transform_new(const Eigen::MatrixXd& X_new, std::size_t k) const;

    // Reconstruct from reduced coordinates Z (approximate inverse)
    Eigen::MatrixXd reconstruct(const Eigen::MatrixXd& Z, std::size_t k) const;

    // Accessors
    const Eigen::MatrixXd& U() const { return U_; }
    const Eigen::VectorXd& S() const { return S_; }
    const Eigen::MatrixXd& V() const { return V_; }

private:
    Eigen::MatrixXd U_;   // m×r
    Eigen::VectorXd S_;   // r
    Eigen::MatrixXd V_;   // n×r
};