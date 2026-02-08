#pragma once

#include <vector>
#include <cstddef>

#include "Kernel/Kernel.hpp"
#include "Kernel/kernel_cache.hpp"

namespace mlpp::classifiers
{

/**
 * @brief Support Vector Machine for binary classification.
 *
 * This class implements a soft-margin kernel SVM in dual form.
 * Training is performed by optimizing the dual variables subject
 * to box constraints and equality constraints.
 */
class SVM
{
public:
    /**
     * @brief Construct an SVM with a given kernel and regularization.
     *
     * @param kernel Kernel function
     * @param C      Regularization parameter
     * @param tol    Numerical tolerance
     */
    explicit SVM(
        kernel::Kernel kernel,
        double C = 1.0,
        double tol = 1e-3
    );

    /**
     * @brief Fit the model to training data.
     *
     * @param X Feature vectors
     * @param y Labels (+1 or -1)
     */
    void fit(
        const std::vector<std::vector<double>>& X,
        const std::vector<int>& y
    );

    /**
     * @brief Predict the label of a single sample.
     */
    int predict(const std::vector<double>& x) const;

    /**
     * @brief Compute the decision function value.
     */
    double decision_function(const std::vector<double>& x) const;

private:
    // Training data
    std::vector<std::vector<double>> X_;
    std::vector<int> y_;

    // Dual variables
    std::vector<double> alpha_;

    // Bias term
    double b_{0.0};

    // Hyperparameters
    double C_;
    double tol_;

    // Kernel infrastructure
    kernel::Kernel kernel_;
    kernel::KernelCache kernel_cache_;

private:
    /**
     * @brief Initialize dual variables and kernel cache.
     */
    void initialize();

    /**
     * @brief Compute decision function on training sample i.
     */
    double decision_function(std::size_t i) const;
};

} // namespace mlpp::classifiers
