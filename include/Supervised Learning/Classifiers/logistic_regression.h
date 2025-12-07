#pragma once

#include <vector>
#include <cstddef>
#include <stdexcept>
#include <cmath>

template <typename T>
class LogisticRegression {
public:
    LogisticRegression(T learning_rate = static_cast<T>(0.01),
                       std::size_t max_iterations = 1000,
                       bool fit_intercept = true);

    void fit(const std::vector<std::vector<T>>& input_features,
             const std::vector<T>& target_values);

    std::vector<T> predict_proba(const std::vector<std::vector<T>>& input_features) const;
    std::vector<int> predict(const std::vector<std::vector<T>>& input_features) const;

    const std::vector<T>& get_weights() const { return weights; }
    T get_intercept() const { return intercept; }

private:
    std::vector<T> weights;   
    T intercept;              

    T learning_rate;
    std::size_t max_iterations;
    bool fit_intercept;

    T sigmoid(T z) const;
    std::vector<T> compute_gradient(const std::vector<std::vector<T>>& input_features,
                                    const std::vector<T>& target_values) const;

    T compute_logit(const std::vector<T>& x) const;
};

#include "logistic_regression.inl"
