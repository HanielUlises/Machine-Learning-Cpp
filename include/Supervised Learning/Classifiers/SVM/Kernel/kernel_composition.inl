#pragma once
#include <cmath>

#include "kernel_composition.hpp"

namespace mlpp::classifiers::kernel
{

inline SumKernel::SumKernel(KernelFunction k1,
                            KernelFunction k2) noexcept
    : k1_(std::move(k1)),
      k2_(std::move(k2))
{}

inline double
SumKernel::operator()(const Vector& x,
                      const Vector& y) const noexcept
{
    return k1_(x, y) + k2_(x, y);
}

inline std::unique_ptr<Kernel>
SumKernel::clone() const
{
    return std::make_unique<SumKernel>(*this);
}

inline ProductKernel::ProductKernel(KernelFunction k1,
                                    KernelFunction k2) noexcept
    : k1_(std::move(k1)),
      k2_(std::move(k2))
{}

inline double
ProductKernel::operator()(const Vector& x,
                          const Vector& y) const noexcept
{
    return k1_(x, y) * k2_(x, y);
}

inline std::unique_ptr<Kernel>
ProductKernel::clone() const
{
    return std::make_unique<ProductKernel>(*this);
}

inline ScaledKernel::ScaledKernel(double scale,
                                  KernelFunction kernel) noexcept
    : scale_(scale),
      kernel_(std::move(kernel))
{}

inline double
ScaledKernel::operator()(const Vector& x,
                         const Vector& y) const noexcept
{
    return scale_ * kernel_(x, y);
}

inline std::unique_ptr<Kernel>
ScaledKernel::clone() const
{
    return std::make_unique<ScaledKernel>(*this);
}

inline ExponentialKernel::ExponentialKernel(KernelFunction base) noexcept
    : base_(std::move(base))
{}

inline double
ExponentialKernel::operator()(const Vector& x,
                              const Vector& y) const noexcept
{
    return std::exp(base_(x, y));
}

inline std::unique_ptr<Kernel>
ExponentialKernel::clone() const
{
    return std::make_unique<ExponentialKernel>(*this);
}

} // namespace mlpp::classifiers::kernel
