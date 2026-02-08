#pragma once

#include "kernel_cache.hpp"

namespace mlpp::classifiers::kernel
{

inline KernelCache::KernelCache(const std::vector<Vector>& data,
                                KernelFunction kernel)
    : data_(data),
      kernel_(std::move(kernel)),
      gram_(data.size(), data.size()),
      computed_(data.size(), data.size())
{
    gram_.setZero();
    computed_.setConstant(false);
}

inline std::size_t
KernelCache::size() const noexcept
{
    return data_.size();
}

inline void
KernelCache::compute_entry(std::size_t i,
                           std::size_t j) const
{
    const double value = kernel_(data_[i], data_[j]);

    gram_(i, j) = value;
    gram_(j, i) = value;

    computed_(i, j) = true;
    computed_(j, i) = true;
}

inline double
KernelCache::operator()(std::size_t i,
                         std::size_t j) const
{
    if (!computed_(i, j))
        compute_entry(i, j);

    return gram_(i, j);
}

inline void
KernelCache::precompute() const
{
    const std::size_t n = size();

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = i; j < n; ++j)
        {
            if (!computed_(i, j))
                compute_entry(i, j);
        }
    }
}

inline const KernelCache::Matrix&
KernelCache::gram_matrix() const
{
    precompute();
    return gram_;
}

inline const KernelFunction&
KernelCache::kernel() const noexcept
{
    return kernel_;
}

} // namespace mlpp::classifiers::kernel
