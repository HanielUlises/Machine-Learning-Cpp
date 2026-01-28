#pragma once
#include "Kernel.hpp"

#include <cstddef>
#include <memory>

namespace mlpp::classifiers::kernel
{

/**
 * Linear kernel
 *
 * k(x, y) = <x, y>
 *
 * Corresponds to the identity feature map in ℝⁿ.
 */
class LinearKernel final : public Kernel
{
public:
    [[nodiscard]]
    double operator()(const Vector& x,
                      const Vector& y) const noexcept override;

    [[nodiscard]]
    std::unique_ptr<Kernel> clone() const override;
};

/**
 * Polynomial kernel
 *
 * k(x, y) = (γ <x, y> + c)^d
 *
 * Generates a finite-dimensional RKHS
 * (dimension grows combinatorially with d).
 */
class PolynomialKernel final : public Kernel
{
public:
    PolynomialKernel(double gamma,
                     double coef0,
                     std::size_t degree) noexcept;

    [[nodiscard]]
    double operator()(const Vector& x,
                      const Vector& y) const noexcept override;

    [[nodiscard]]
    std::unique_ptr<Kernel> clone() const override;

private:
    double gamma_;
    double coef0_;
    std::size_t degree_;
};

/**
 * Radial Basis Function (Gaussian) kernel
 *
 * k(x, y) = exp(-γ ||x - y||²)
 *
 * Induces an infinite-dimensional, separable RKHS.
 * Universally consistent under mild conditions.
 */
class RBFKernel final : public Kernel
{
public:
    explicit RBFKernel(double gamma) noexcept;

    [[nodiscard]]
    double operator()(const Vector& x,
                      const Vector& y) const noexcept override;

    [[nodiscard]]
    std::unique_ptr<Kernel> clone() const override;

private:
    double gamma_;
};

} // namespace mlpp::classifiers::kernel

#include "rkhs_kernels.inl"
