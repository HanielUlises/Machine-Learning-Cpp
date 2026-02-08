#pragma once

#include <utility>
#include <cstddef>

namespace mlpp::classifiers
{

inline SVM::SVM(
    const kernel::Kernel& kernel,
    double C,
    double tol
)
    : C_(C),
      tol_(tol),
      kernel_(kernel.clone())
{
}

inline void SVM::fit(
    const std::vector<std::vector<double>>& X,
    const std::vector<int>& y
)
{
    X_ = X;
    y_ = y;

    initialize();
}

inline void SVM::initialize()
{
    const std::size_t n = X_.size();

    alpha_.assign(n, 0.0);
    b_ = 0.0;

    kernel_cache_.emplace(X_, *kernel_);
}

inline double SVM::decision_function(std::size_t i) const
{
    double value = b_;

    const std::size_t n = alpha_.size();
    for (std::size_t j = 0; j < n; ++j)
    {
        if (alpha_[j] != 0.0)
        {
            value += alpha_[j]
                   * static_cast<double>(y_[j])
                   * (*kernel_cache_)(j, i);
        }
    }

    return value;
}

inline double SVM::decision_function(
    const std::vector<double>& x
) const
{
    double value = b_;

    const std::size_t n = alpha_.size();
    for (std::size_t i = 0; i < n; ++i)
    {
        if (alpha_[i] != 0.0)
        {
            value += alpha_[i]
                   * static_cast<double>(y_[i])
                   * (*kernel_)(X_[i], x);
        }
    }

    return value;
}

inline int SVM::predict(
    const std::vector<double>& x
) const
{
    return decision_function(x) >= 0.0 ? 1 : -1;
}

} // namespace mlpp::classifiers
