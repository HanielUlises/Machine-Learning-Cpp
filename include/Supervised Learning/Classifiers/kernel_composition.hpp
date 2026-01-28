#pragma once
#include "Kernel.hpp"

#include <memory>
#include <utility>

namespace mlpp::classifiers::kernel
{

/**
 * Sum of two kernels
 *
 * k(x, y) = k₁(x, y) + k₂(x, y)
 *
 * The resulting kernel is positive semi-definite
 * if both k₁ and k₂ are PSD.
 *
 * Corresponds to the direct sum of RKHSs:
 *   H = H₁ ⊕ H₂
 */
class SumKernel final : public Kernel
{
public:
    SumKernel(KernelFunction k1,
              KernelFunction k2) noexcept;

    [[nodiscard]]
    double operator()(const Vector& x,
                      const Vector& y) const noexcept override;

    [[nodiscard]]
    std::unique_ptr<Kernel> clone() const override;

private:
    KernelFunction k1_;
    KernelFunction k2_;
};

/**
 * Product of two kernels
 *
 * k(x, y) = k₁(x, y) · k₂(x, y)
 *
 * Preserves positive semi-definiteness.
 * Induces the tensor product RKHS.
 */
class ProductKernel final : public Kernel
{
public:
    ProductKernel(KernelFunction k1,
                  KernelFunction k2) noexcept;

    [[nodiscard]]
    double operator()(const Vector& x,
                      const Vector& y) const noexcept override;

    [[nodiscard]]
    std::unique_ptr<Kernel> clone() const override;

private:
    KernelFunction k1_;
    KernelFunction k2_;
};

/**
 * Scaled kernel
 *
 * k(x, y) = α · k(x, y),   α ≥ 0
 *
 * Scalar multiplication preserves PSD.
 */
class ScaledKernel final : public Kernel
{
public:
    ScaledKernel(double scale,
                 KernelFunction kernel) noexcept;

    [[nodiscard]]
    double operator()(const Vector& x,
                      const Vector& y) const noexcept override;

    [[nodiscard]]
    std::unique_ptr<Kernel> clone() const override;

private:
    double scale_;
    KernelFunction kernel_;
};

/**
 * Exponentiated kernel
 *
 * k(x, y) = exp( k₀(x, y) )
 *
 * Useful in practice, but only PSD if k₀
 * is conditionally PSD.
 *
 * Provided as an advanced construction.
 */
class ExponentialKernel final : public Kernel
{
public:
    explicit ExponentialKernel(KernelFunction base) noexcept;

    [[nodiscard]]
    double operator()(const Vector& x,
                      const Vector& y) const noexcept override;

    [[nodiscard]]
    std::unique_ptr<Kernel> clone() const override;

private:
    KernelFunction base_;
};

} // namespace mlpp::classifiers::kernel
