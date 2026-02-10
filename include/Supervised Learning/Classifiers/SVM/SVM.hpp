#pragma once

#include "Kernel/kernel_cache.hpp"

#include <Eigen/Dense>
#include <vector>
#include <cstddef>

namespace mlpp::classifiers::kernel
{

/**
 * Support Vector Machine (binary, kernelized)
 *
 * This class represents the dual formulation:
 *
 *   maximize   W(α) = Σ α_i − 1/2 Σ Σ α_i α_j y_i y_j K(x_i, x_j)
 *
 *   subject to:
 *     0 ≤ α_i ≤ C
 *     Σ α_i y_i = 0
 *
 * The optimization procedure is intentionally separated
 * from the mathematical structure.
 */
class SVM
{
public:
    using LabelVector = Eigen::VectorXd;
    using AlphaVector = Eigen::VectorXd;

    /**
     * Construct an SVM model.
     *
     * @param data   Training samples
     * @param labels Class labels in {−1, +1}
     * @param kernel Kernel function
     * @param C      Soft margin penalty parameter
     */
    SVM(const std::vector<Vector>& data,
        LabelVector labels,
        KernelFunction kernel,
        double C);

    /**
     * Train the SVM model.
     *
     * Optimization strategy is implementation-defined.
     */
    void fit();

    /**
     * Evaluate the decision function:
     *
     *   f(x) = Σ α_i y_i K(x_i, x) + b
     */
    [[nodiscard]]
    double decision(const Vector& x) const;

    /**
     * Predict class label.
     */
    [[nodiscard]]
    int predict(const Vector& x) const;

    // Indices i such that α_i > 0
    [[nodiscard]]
    std::vector<std::size_t> support_indices(double eps = 1e-8) const;


private:
    /**
     * Compute bias term using support vectors.
     */
    void compute_bias();

private:
    const std::vector<Vector>& data_;
    LabelVector labels_;
    Eigen::VectorXd error_;

    double C_;

    KernelCache kernel_cache_;

    AlphaVector alpha_;
    double bias_;
};

} // namespace mlpp::classifiers::kernel

#include "SVM.inl"
