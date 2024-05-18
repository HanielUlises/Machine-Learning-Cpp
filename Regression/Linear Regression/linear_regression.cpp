#include "linear_regression.h"
#include <iostream>
#include <numeric>
#include <stdexcept>

// Constructor to initialize the LinearRegression object with default slope and intercept
LinearRegression::LinearRegression() : slope(0.0), intercept(0.0) {}

// Function to fit the linear regression model to the provided data
void LinearRegression::fit(const std::vector<double>& X, const std::vector<double>& y) {
    // Ensure the input vectors X and y have the same size
    if (X.size() != y.size()) {
        throw std::invalid_argument("The size of X and y must be equal.");
    }

    // Compute the means of X and y
    double X_mean = mean(X);
    double y_mean = mean(y);
    
    // Calculate the slope and intercept using the least squares method
    slope = covariance(X, y) / variance(X);
    intercept = y_mean - slope * X_mean;
}

// Function to predict the values of y based on the fitted model and input vector X
std::vector<double> LinearRegression::predict(const std::vector<double>& X) const {
    std::vector<double> predictions;
    for (const auto& x : X) {
        predictions.push_back(intercept + slope * x);
    }
    return predictions;
}

// Function to calculate the coefficient of determination (R² score) for the model
double LinearRegression::score(const std::vector<double>& X, const std::vector<double>& y) const {
    // Ensure the input vectors X and y have the same size
    if (X.size() != y.size()) {
        throw std::invalid_argument("The size of X and y must be equal.");
    }

    double y_mean = mean(y);
    std::vector<double> y_pred = predict(X);
    double ss_total = 0.0;
    double ss_res = 0.0;

    // Compute the total sum of squares and the residual sum of squares
    for (size_t i = 0; i < y.size(); ++i) {
        ss_total += (y[i] - y_mean) * (y[i] - y_mean);
        ss_res += (y[i] - y_pred[i]) * (y[i] - y_pred[i]);
    }

    // Return the R² score
    return 1 - (ss_res / ss_total);
}

// Function to plot the regression line along with the original data points
void LinearRegression::plot_regression_line(const std::vector<double>& X, const std::vector<double>& y, const std::string& title) const {
    // Generate predicted values
    std::vector<double> y_pred = predict(X);
    
    // Original data points
    plt::scatter(X, y);
    
    // Regression line
    plt::plot(X, y_pred);
    
    plt::title(title);
    plt::xlabel("X");
    plt::ylabel("y");
    
    plt::show();
}

// Function to calculate the mean of a vector
double LinearRegression::mean(const std::vector<double>& v) const {
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

// Function to calculate the covariance between two vectors
double LinearRegression::covariance(const std::vector<double>& X, const std::vector<double>& y) const {
    double X_mean = mean(X);
    double y_mean = mean(y);
    double cov = 0.0;

    // Compute the covariance
    for (size_t i = 0; i < X.size(); ++i) {
        cov += (X[i] - X_mean) * (y[i] - y_mean);
    }

    return cov / X.size();
}

// Function to calculate the variance of a vector
double LinearRegression::variance(const std::vector<double>& v) const {
    double v_mean = mean(v);
    double var = 0.0;

    // Compute the variance
    for (const auto& val : v) {
        var += (val - v_mean) * (val - v_mean);
    }

    return var / v.size();
}
