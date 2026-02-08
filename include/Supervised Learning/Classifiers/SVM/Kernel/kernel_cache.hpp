#pragma once

#include "kernel.hpp"

#include <Eigen/Dense>
#include <vector>
#include <cstddef>

namespace mlpp::classifiers::kernel
{

/**
 * @brief Kernel (Gram) matrix cache.
 *
 * This class stores and manages evaluations of the kernel-induced
 * Gram matrix
 *
 *     K_{ij} = k(x_i, x_j),
 *
 * for a fixed dataset {x_i} and a fixed kernel function k.
 *
 * The cache supports lazy evaluation, symmetric storage, and
 * full precomputation. Internally, evaluations are stored in
 * an Eigen dense matrix for efficient numerical access.
 */
class KernelCache
{
public:
    using Matrix = Eigen::MatrixXd;

    /**
     * @brief Construct a kernel cache for a dataset and kernel.
     *
     * @param data    Input samples
     * @param kernel  Kernel function
     */
    KernelCache(const std::vector<Vector>& data,
                KernelFunction kernel);

    /**
     * @brief Return the number of samples.
     */
    [[nodiscard]]
    std::size_t size() const noexcept;

    /**
     * @brief Access a single Gram matrix entry.
     *
     * Evaluates K_{ij} = k(x_i, x_j) if it has not been computed yet.
     */
    [[nodiscard]]
    double operator()(std::size_t i,
                      std::size_t j) const;

    /**
     * @brief Access the full Gram matrix.
     *
     * Ensures that all entries are computed before returning.
     */
    [[nodiscard]]
    const Matrix& gram_matrix() const;

    /**
     * @brief Force computation of all kernel evaluations.
     */
    void precompute() const;

    /**
     * @brief Access the underlying kernel function.
     */
    [[nodiscard]]
    const KernelFunction& kernel() const noexcept;

private:
    /**
     * @brief Compute and store a single Gram matrix entry.
     *
     * Exploits kernel symmetry:
     *   K_{ij} = K_{ji}.
     */
    void compute_entry(std::size_t i,
                       std::size_t j) const;

private:
    const std::vector<Vector>& data_;
    KernelFunction kernel_;

    mutable Matrix gram_;
    mutable Eigen::ArrayXX<bool> computed_;
};

} // namespace mlpp::classifiers::kernel

#include "kernel_cache.inl"
