#include "linear_regression.h"
#include <stdexcept>
#include <numeric>
#include <cmath>
#include <algorithm>

template<typename T>
LinearRegression<T>::LinearRegression(bool fit_intercept, T regularization_param) 
    : intercept(0.0), fit_intercept(fit_intercept), regularization_param(regularization_param) {}

// Fit the linear regression model to the provided input features and target variable
template<typename T>
void LinearRegression<T>::fit(const std::vector<std::vector<T>>& input_features, const std::vector<T>& target_variable) {
    if (input_features.size() != target_variable.size()) {
        throw std::invalid_argument("The size of input_features and target_variable must be equal.");
    }

    size_t n_samples = input_features.size();
    size_t n_features = input_features[0].size();
    
    std::vector<std::vector<T>> augmented_features = input_features;

    if (fit_intercept) {
        augmented_features = add_bias_term(input_features);
    }

    // Normalize the input features
    augmented_features = normalize(augmented_features);

    coefficients.resize(n_features + (fit_intercept ? 1 : 0), 0.0);
    
    // Compute gradient for coefficient update
    std::vector<T> gradient = compute_gradient(augmented_features, target_variable);
    
    // Update coefficients with regularization
    for (size_t i = 0; i < coefficients.size(); ++i) {
        coefficients[i] -= regularization_param * gradient[i];
    }

    if (fit_intercept) {
        intercept = coefficients[0];
        coefficients.erase(coefficients.begin()); // Remove the intercept from coefficients
    }
}

// Predict target values based on the fitted model and new input features
template<typename T>
std::vector<T> LinearRegression<T>::predict(const std::vector<std::vector<T>>& input_features) const {
    std::vector<std::vector<T>> augmented_features = input_features;

    if (fit_intercept) {
        augmented_features = add_bias_term(input_features);
    }

    std::vector<T> predictions;
    for (const auto& row : augmented_features) {
        T prediction = intercept;
        for (size_t i = 0; i < coefficients.size(); ++i) {
            prediction += coefficients[i] * row[i];
        }
        predictions.push_back(prediction);
    }
    return predictions;
}

// Calculate the coefficient of determination (R² score) for the model
template<typename T>
T LinearRegression<T>::score(const std::vector<std::vector<T>>& input_features, const std::vector<T>& target_variable) const {
    if (input_features.size() != target_variable.size()) {
        throw std::invalid_argument("The size of input_features and target_variable must be equal.");
    }

    T target_mean = mean(target_variable);
    std::vector<T> predicted_values = predict(input_features);
    T total_sum_squares = 0.0;
    T residual_sum_squares = 0.0;

    for (size_t i = 0; i < target_variable.size(); ++i) {
        total_sum_squares += (target_variable[i] - target_mean) * (target_variable[i] - target_mean);
        residual_sum_squares += (target_variable[i] - predicted_values[i]) * (target_variable[i] - predicted_values[i]);
    }

    return 1 - (residual_sum_squares / total_sum_squares);
}

// Calculate the mean of a vector
template<typename T>
T LinearRegression<T>::mean(const std::vector<T>& values) const {
    return std::accumulate(values.begin(), values.end(), static_cast<T>(0.0)) / values.size();
}

// Calculate the variance of a vector
template<typename T>
T LinearRegression<T>::variance(const std::vector<T>& values) const {
    T mean_value = mean(values);
    T var = 0.0;

    for (const auto& val : values) {
        var += (val - mean_value) * (val - mean_value);
    }

    return var / values.size();
}

// Calculate the covariance between two vectors
template<typename T>
T LinearRegression<T>::covariance(const std::vector<T>& feature_values, const std::vector<T>& target_values) const {
    T feature_mean = mean(feature_values);
    T target_mean = mean(target_values);
    T cov = 0.0;

    for (size_t i = 0; i < feature_values.size(); ++i) {
        cov += (feature_values[i] - feature_mean) * (target_values[i] - target_mean);
    }

    return cov / feature_values.size();
}

// Compute gradient for gradient descent-based fitting
template<typename T>
std::vector<T> LinearRegression<T>::compute_gradient(const std::vector<std::vector<T>>& input_features, const std::vector<T>& target_values) const {
    std::vector<T> gradient(coefficients.size(), 0.0);

    for (size_t i = 0; i < input_features.size(); ++i) {
        T error = target_values[i] - predict({input_features[i]})[0];
        for (size_t j = 0; j < coefficients.size(); ++j) {
            gradient[j] += -2 * error * input_features[i][j];
        }
    }

    return gradient;
}

// Add a bias term to the input features
template<typename T>
std::vector<std::vector<T>> LinearRegression<T>::add_bias_term(const std::vector<std::vector<T>>& input_features) const {
    std::vector<std::vector<T>> augmented_features(input_features.size(), std::vector<T>(input_features[0].size() + 1, 1.0));
    
    for (size_t i = 0; i < input_features.size(); ++i) {
        for (size_t j = 0; j < input_features[i].size(); ++j) {
            augmented_features[i][j + 1] = input_features[i][j];
        }
    }
    
    return augmented_features;
}

// Normalization of input features
template<typename T>
std::vector<std::vector<T>> LinearRegression<T>::normalize(const std::vector<std::vector<T>>& input_features) const {
    std::vector<std::vector<T>> normalized_features = input_features;
    size_t n_features = input_features[0].size();

    for (size_t j = 0; j < n_features; ++j) {
        T mean_value = 0.0;
        T std_dev = 0.0;

        // Calculate mean for the feature
        for (size_t i = 0; i < input_features.size(); ++i) {
            mean_value += input_features[i][j];
        }
        mean_value /= input_features.size();

        // Calculate standard deviation for the feature
        for (size_t i = 0; i < input_features.size(); ++i) {
            std_dev += std::pow(input_features[i][j] - mean_value, 2);
        }
        std_dev = std::sqrt(std_dev / input_features.size());

        // Normalize the feature values
        for (size_t i = 0; i < input_features.size(); ++i) {
            normalized_features[i][j] = (input_features[i][j] - mean_value) / std_dev;
        }
    }

    return normalized_features;
}

// Getters
template<typename T>
std::vector<T> LinearRegression<T>::get_coefficients() const {
    return coefficients;
}

template<typename T>
T LinearRegression<T>::get_intercept() const {
    return intercept;
}

template class LinearRegression<double>;
template class LinearRegression<float>;
