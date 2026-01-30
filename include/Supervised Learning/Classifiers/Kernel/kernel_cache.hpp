#pragma once

#include "Kernel.hpp"

#include <Eigen/Dense>
#include <vector>
#include <cstddef>

namespace mlpp::classifiers::kernel
{

/**
 * Kernel (Gram) matrix cache
 *
 * Stores and manages evaluations:
 *   K_ij = k(x_i, x_j)
 *
 * Supports:
 *  - lazy evaluation
 *  - full precomputation
 *  - Eigen-accelerated access
 */
class KernelCache
{
public:
    using Matrix = Eigen::MatrixXd;

    KernelCache(const std::vector<Vector>& data,
                KernelFunction kernel);

    [[nodiscard]]
    std::size_t size() const noexcept;


    [[nodiscard]]
    double operator()(std::size_t i,
                      std::size_t j);

    [[nodiscard]]
    const Matrix& gram_matrix();

    void precompute();
    
    [[nodiscard]]
    const KernelFunction& kernel() const noexcept;

private:
    void compute_entry(std::size_t i,
                       std::size_t j);

private:
    const std::vector<Vector>& data_;
    KernelFunction kernel_;

    Matrix gram_;
    Eigen::ArrayXX<bool> computed_;
};

} // namespace mlpp::classifiers::kernel

#include "kernel_cache.inl"
