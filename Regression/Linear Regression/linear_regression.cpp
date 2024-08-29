#include "linear_regression.h"
#include <stdexcept>
#include <numeric>
#include <cmath>
#include <algorithm>

template<typename T>
LinearRegression<T>::LinearRegression(bool fit_intercept, T regularization_param) 
    : intercept(0.0), fit_intercept(fit_intercept), regularization_param(regularization_param) {}

// Function to fit the linear regression model to the provided data
template<typename T>
void LinearRegression<T>::fit(const std::vector<std::vector<T>>& X, const std::vector<T>& y) {
    if (X.size() != y.size()) {
        throw std::invalid_argument("The size of X and y must be equal.");
    }

    size_t n_samples = X.size();
    size_t n_features = X[0].size();
    
    std::vector<std::vector<T>> X_augmented = X;

    if (fit_intercept) {
        X_augmented = add_bias_term(X);
    }

    // Normalization (Optional step, depending on data characteristics)
    X_augmented = normalize(X_augmented);

    coefficients.resize(n_features + (fit_intercept ? 1 : 0), 0.0);
    
    std::vector<T> gradient = compute_gradient(X_augmented, y);
    
    // Updating coefficients using the normal equation
    for (size_t i = 0; i < coefficients.size(); ++i) {
        coefficients[i] -= regularization_param * gradient[i];
    }

    if (fit_intercept) {
        intercept = coefficients[0];
        // The intercept term is removed from coefficients vector
        coefficients.erase(coefficients.begin()); 
    }
}

// Function to predict the values of y based on the fitted model and input vector X
template<typename T>
std::vector<T> LinearRegression<T>::predict(const std::vector<std::vector<T>>& X) const {
    std::vector<std::vector<T>> X_augmented = X;

    if (fit_intercept) {
        X_augmented = add_bias_term(X);
    }

    std::vector<T> predictions;
    for (const auto& row : X_augmented) {
        T prediction = intercept;
        for (size_t i = 0; i < coefficients.size(); ++i) {
            prediction += coefficients[i] * row[i];
        }
        predictions.push_back(prediction);
    }
    return predictions;
}

// Function to calculate the coefficient of determination (R² score) for the model
template<typename T>
T LinearRegression<T>::score(const std::vector<std::vector<T>>& X, const std::vector<T>& y) const {
    if (X.size() != y.size()) {
        throw std::invalid_argument("The size of X and y must be equal.");
    }

    T y_mean = mean(y);
    std::vector<T> y_pred = predict(X);
    T ss_total = 0.0;
    T ss_res = 0.0;

    for (size_t i = 0; i < y.size(); ++i) {
        ss_total += (y[i] - y_mean) * (y[i] - y_mean);
        ss_res += (y[i] - y_pred[i]) * (y[i] - y_pred[i]);
    }

    return 1 - (ss_res / ss_total);
}

// Function to calculate the mean of a vector
template<typename T>
T LinearRegression<T>::mean(const std::vector<T>& v) const {
    return std::accumulate(v.begin(), v.end(), static_cast<T>(0.0)) / v.size();
}

// Function to calculate the variance of a vector
template<typename T>
T LinearRegression<T>::variance(const std::vector<T>& v) const {
    T v_mean = mean(v);
    T var = 0.0;

    for (const auto& val : v) {
        var += (val - v_mean) * (val - v_mean);
    }

    return var / v.size();
}

// Function to calculate the covariance between two vectors
template<typename T>
T LinearRegression<T>::covariance(const std::vector<T>& X, const std::vector<T>& y) const {
    T X_mean = mean(X);
    T y_mean = mean(y);
    T cov = 0.0;

    for (size_t i = 0; i < X.size(); ++i) {
        cov += (X[i] - X_mean) * (y[i] - y_mean);
    }

    return cov / X.size();
}

// Function to compute gradient (for gradient descent-based fitting)
template<typename T>
std::vector<T> LinearRegression<T>::compute_gradient(const std::vector<std::vector<T>>& X, const std::vector<T>& y) const {
    std::vector<T> gradient(coefficients.size(), 0.0);

    for (size_t i = 0; i < X.size(); ++i) {
        T error = y[i] - predict({X[i]})[0];
        for (size_t j = 0; j < coefficients.size(); ++j) {
            gradient[j] += -2 * error * X[i][j];
        }
    }

    return gradient;
}

// Function to add a bias term (for intercept)
template<typename T>
std::vector<std::vector<T>> LinearRegression<T>::add_bias_term(const std::vector<std::vector<T>>& X) const {
    std::vector<std::vector<T>> X_augmented(X.size(), std::vector<T>(X[0].size() + 1, 1.0));
    
    for (size_t i = 0; i < X.size(); ++i) {
        for (size_t j = 0; j < X[i].size(); ++j) {
            X_augmented[i][j + 1] = X[i][j];
        }
    }
    
    return X_augmented;
}

// Function to normalize data
template<typename T>
std::vector<std::vector<T>> LinearRegression<T>::normalize(const std::vector<std::vector<T>>& X) const {
    std::vector<std::vector<T>> X_normalized = X;
    size_t n_features = X[0].size();

    for (size_t j = 0; j < n_features; ++j) {
        T mean_val = 0.0;
        T std_dev = 0.0;

        for (size_t i = 0; i < X.size(); ++i) {
            mean_val += X[i][j];
        }
        mean_val /= X.size();

        for (size_t i = 0; i < X.size(); ++i) {
            std_dev += std::pow(X[i][j] - mean_val, 2);
        }
        std_dev = std::sqrt(std_dev / X.size());

        for (size_t i = 0; i < X.size(); ++i) {
            X_normalized[i][j] = (X[i][j] - mean_val) / std_dev;
        }
    }

    return X_normalized;
}

// Getter for coefficients
template<typename T>
std::vector<T> LinearRegression<T>::get_coefficients() const {
    return coefficients;
}

// Getter for intercept
template<typename T>
T LinearRegression<T>::get_intercept() const {
    return intercept;
}

// Template specialization is required to avoid linker errors
template class LinearRegression<double>;
template class LinearRegression<float>;

