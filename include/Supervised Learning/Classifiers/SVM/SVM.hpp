#pragma once

#include <vector>
#include <memory>
#include <cstddef>

#include "Kernel/Kernel.hpp"
#include "Kernel/kernel_cache.hpp"

namespace mlpp::classifiers
{

class SVM
{
public:
    /**
     * @brief Construct an SVM with a kernel.
     */
    explicit SVM(
        const kernel::Kernel& kernel,
        double C = 1.0,
        double tol = 1e-3
    );

    void fit(
        const std::vector<std::vector<double>>& X,
        const std::vector<int>& y
    );

    int predict(const std::vector<double>& x) const;

    double decision_function(const std::vector<double>& x) const;

private:
    // Training data
    std::vector<std::vector<double>> X_;
    std::vector<int> y_;

    // Dual variables
    std::vector<double> alpha_;
    double b_{0.0};

    // Hyperparameters
    double C_;
    double tol_;

    // Kernel (polymorphic ownership)
    std::unique_ptr<kernel::Kernel> kernel_;
    kernel::KernelCache kernel_cache_;

private:
    void initialize();

    double decision_function(std::size_t i) const;
};

} // namespace mlpp::classifiers
