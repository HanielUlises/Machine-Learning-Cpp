#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include <vector>

// Template class for linear regression models.
template<typename T>
class LinearRegression {
public:
    // Constructor to initialize parameters for the linear regression model.
    LinearRegression(bool fit_intercept = true, T regularization_param = 0.0);

    // Fits the model to the provided input features and target variable.
    void fit(const std::vector<std::vector<T>>& input_features, const std::vector<T>& target_variable);

    // Predicts target values based on the fitted model and new input features.
    std::vector<T> predict(const std::vector<std::vector<T>>& input_features) const;

    // Calculates the coefficient of determination (R² score) for the model.
    T score(const std::vector<std::vector<T>>& input_features, const std::vector<T>& target_variable) const;

    // Getter function to retrieve the model coefficients.
    std::vector<T> get_coefficients() const;

    // Getter function to retrieve the model intercept.
    T get_intercept() const;

private:
    // Model parameters
    std::vector<T> coefficients;          // Coefficients of the linear model
    T intercept;                          // Intercept of the linear model
    bool fit_intercept;                   // Flag to indicate if intercept should be fitted
    T regularization_param;               // Regularization parameter for the model

    // Utility functions for calculations
    T mean(const std::vector<T>& values) const; // Computes the mean of a vector
    T variance(const std::vector<T>& values) const; // Computes the variance of a vector
    T covariance(const std::vector<T>& feature_values, const std::vector<T>& target_values) const; // Computes covariance
    std::vector<T> compute_gradient(const std::vector<std::vector<T>>& input_features, const std::vector<T>& target_values) const; // Computes the gradient for optimization
    std::vector<std::vector<T>> add_bias_term(const std::vector<std::vector<T>>& input_features) const; // Adds bias term to input features
    std::vector<std::vector<T>> normalize(const std::vector<std::vector<T>>& input_features) const; // Normalizes the input features
};

// Include the implementation file to avoid linker issues with template classes.
#include "linear_regression.cpp"

#endif // LINEAR_REGRESSION_H
