#pragma once

#include "kernel.hpp"

namespace mlpp::classifiers::kernel
{

inline KernelFunction::KernelFunction(const KernelFunction& other)
{
    if (other.impl_)
        impl_ = other.impl_->clone();
}

inline KernelFunction&
KernelFunction::operator=(const KernelFunction& other)
{
    if (this != &other)
    {
        impl_.reset();
        if (other.impl_)
            impl_ = other.impl_->clone();
    }
    return *this;
}

inline KernelFunction::KernelFunction(std::unique_ptr<Kernel> impl)
    : impl_(std::move(impl))
{}

inline double
KernelFunction::operator()(const Vector& x,
                           const Vector& y) const noexcept
{
    return (*impl_)(x, y);
}

inline bool
KernelFunction::valid() const noexcept
{
    return static_cast<bool>(impl_);
}

} // namespace mlpp::classifiers::kernel
