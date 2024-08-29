#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include <vector>

template<typename T>
class LinearRegression {
public:
    LinearRegression(bool fit_intercept = true, T regularization_param = 0.0);

    // Fit the model to the provided data (X: input features, y: target variable)
    void fit(const std::vector<std::vector<T>>& X, const std::vector<T>& y);

    // Predict values based on the fitted model
    std::vector<T> predict(const std::vector<std::vector<T>>& X) const;

    // Calculate the coefficient of determination (R² score) for the model
    T score(const std::vector<std::vector<T>>& X, const std::vector<T>& y) const;

    // Getter functions for slope and intercept
    std::vector<T> get_coefficients() const;
    T get_intercept() const;

private:
    std::vector<T> coefficients;
    T intercept;
    bool fit_intercept;
    T regularization_param;

    // Utility functions
    T mean(const std::vector<T>& v) const;
    T variance(const std::vector<T>& v) const;
    T covariance(const std::vector<T>& X, const std::vector<T>& y) const;
    std::vector<T> compute_gradient(const std::vector<std::vector<T>>& X, const std::vector<T>& y) const;
    std::vector<std::vector<T>> add_bias_term(const std::vector<std::vector<T>>& X) const;
    std::vector<std::vector<T>> normalize(const std::vector<std::vector<T>>& X) const;
};

#include "linear_regression.cpp"

#endif // LINEAR_REGRESSION_H
