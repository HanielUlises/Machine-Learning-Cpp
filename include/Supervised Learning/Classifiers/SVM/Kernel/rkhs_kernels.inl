#pragma once

#include <numeric>
#include <cmath>

#include "rkhs_kernels.hpp"

namespace mlpp::classifiers::kernel
{


inline double
LinearKernel::operator()(const Vector& x,
                         const Vector& y) const noexcept
{
    return std::inner_product(x.begin(), x.end(), y.begin(), 0.0);
}

inline std::unique_ptr<Kernel>
LinearKernel::clone() const
{
    return std::make_unique<LinearKernel>(*this);
}

inline PolynomialKernel::PolynomialKernel(double gamma,
                                          double coef0,
                                          std::size_t degree) noexcept
    : gamma_(gamma),
      coef0_(coef0),
      degree_(degree)
{}

inline double
PolynomialKernel::operator()(const Vector& x,
                             const Vector& y) const noexcept
{
    const double dot =
        std::inner_product(x.begin(), x.end(), y.begin(), 0.0);

    return std::pow(
        gamma_ * dot + coef0_,
        static_cast<double>(degree_)
    );
}

inline std::unique_ptr<Kernel>
PolynomialKernel::clone() const
{
    return std::make_unique<PolynomialKernel>(*this);
}

inline RBFKernel::RBFKernel(double gamma) noexcept
    : gamma_(gamma)
{}

inline double
RBFKernel::operator()(const Vector& x,
                      const Vector& y) const noexcept
{
    double squared_distance = 0.0;

    for (std::size_t i = 0; i < x.size(); ++i)
    {
        const double diff = x[i] - y[i];
        squared_distance += diff * diff;
    }

    return std::exp(-gamma_ * squared_distance);
}

inline std::unique_ptr<Kernel>
RBFKernel::clone() const
{
    return std::make_unique<RBFKernel>(*this);
}

} // namespace mlpp::classifiers::kernel
