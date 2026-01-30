#pragma once

#include "../Kernel/Kernel.hpp"
#include "../Kernel/kernel_cache.hpp"

#include <vector>
#include <cstddef>

namespace mlpp::classifiers
{

/**
 * Kernelized Perceptron (dual form)
 *
 * Decision function:
 *
 *   f(x) = sign( sum_i α_i y_i k(x_i, x) )
 *
 */
class KernelPerceptron
{
public:
    KernelPerceptron(const std::vector<kernel::Vector>& X,
                     const std::vector<int>& y,
                     kernel::KernelFunction kernel,
                     std::size_t max_epochs = 100);

    void fit();

    [[nodiscard]]
    int predict(const kernel::Vector& x) const;

    [[nodiscard]]
    std::size_t mistakes() const noexcept;

private:
    const std::vector<kernel::Vector>& X_;
    const std::vector<int>& y_;

    kernel::KernelCache cache_;

    std::vector<double> alpha_;
    std::size_t max_epochs_;
    std::size_t mistakes_;
};

} // namespace mlpp::classifiers
