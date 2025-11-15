#pragma once

#include <vector>

template<typename T>
class LinearRegression {
public:
    LinearRegression(bool fit_intercept = true, T regularization_param = 0.0);

    void fit(const std::vector<std::vector<T>>& input_features,
             const std::vector<T>& target_variable);

    std::vector<T> predict(const std::vector<std::vector<T>>& input_features) const;

    T score(const std::vector<std::vector<T>>& input_features,
            const std::vector<T>& target_variable) const;

    std::vector<T> get_coefficients() const;
    T get_intercept() const;

private:
    std::vector<T> coefficients;
    T intercept;

    bool fit_intercept;
    T regularization_param;

    T mean(const std::vector<T>& values) const;
    T variance(const std::vector<T>& values) const;
    T covariance(const std::vector<T>& feature_values,
                 const std::vector<T>& target_values) const;

    std::vector<T> compute_gradient(const std::vector<std::vector<T>>& input_features,
                                    const std::vector<T>& target_values) const;

    std::vector<std::vector<T>> add_bias_term(const std::vector<std::vector<T>>& input_features) const;
    std::vector<std::vector<T>> normalize(const std::vector<std::vector<T>>& input_features) const;
};

#include "linear_regression.inl"
