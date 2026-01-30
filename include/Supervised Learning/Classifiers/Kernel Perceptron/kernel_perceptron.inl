#pragma once

#include "kernel_perceptron.hpp"

namespace mlpp::classifiers
{

inline KernelPerceptron::KernelPerceptron(
    const std::vector<kernel::Vector>& X,
    const std::vector<int>& y,
    kernel::KernelFunction kernel,
    std::size_t max_epochs)
    : X_(X),
      y_(y),
      cache_(X, std::move(kernel)),
      alpha_(X.size(), 0.0),
      max_epochs_(max_epochs),
      mistakes_(0)
{
}

inline void
KernelPerceptron::fit()
{
    const std::size_t n = X_.size();
    mistakes_ = 0;

    for (std::size_t epoch = 0; epoch < max_epochs_; ++epoch)
    {
        bool any_update = false;

        for (std::size_t i = 0; i < n; ++i)
        {
            double decision = 0.0;

            for (std::size_t j = 0; j < n; ++j)
            {
                if (alpha_[j] != 0.0)
                {
                    decision += alpha_[j] * y_[j] * cache_(j, i);
                }
            }

            if (y_[i] * decision <= 0.0)
            {
                alpha_[i] += 1.0;
                ++mistakes_;
                any_update = true;
            }
        }

        if (!any_update)
            break;
    }
}

inline int
KernelPerceptron::predict(const kernel::Vector& x) const
{
    double sum = 0.0;
    const std::size_t n = X_.size();

    for (std::size_t i = 0; i < n; ++i)
    {
        if (alpha_[i] != 0.0)
        {
            sum += alpha_[i] * y_[i] *
                   cache_.kernel()(X_[i], x);
        }
    }

    return sum >= 0.0 ? 1 : -1;
}

inline std::size_t
KernelPerceptron::mistakes() const noexcept
{
    return mistakes_;
}

} // namespace mlpp::classifiers
