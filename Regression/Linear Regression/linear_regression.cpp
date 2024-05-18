#include "linear_regression.h"
#include <iostream>
#include <numeric>
#include <stdexcept>

LinearRegression::LinearRegression() : slope(0.0), intercept(0.0) {}

void LinearRegression::fit(const std::vector<double>& X, const std::vector<double>& y) {
    if (X.size() != y.size()) {
        throw std::invalid_argument("The size of X and y must be equal.");
    }

    double X_mean = mean(X);
    double y_mean = mean(y);
    
    slope = covariance(X, y) / variance(X);
    intercept = y_mean - slope * X_mean;
}

std::vector<double> LinearRegression::predict(const std::vector<double>& X) const {
    std::vector<double> predictions;
    for (const auto& x : X) {
        predictions.push_back(intercept + slope * x);
    }
    return predictions;
}

double LinearRegression::score(const std::vector<double>& X, const std::vector<double>& y) const {
    if (X.size() != y.size()) {
        throw std::invalid_argument("The size of X and y must be equal.");
    }

    double y_mean = mean(y);
    std::vector<double> y_pred = predict(X);
    double ss_total = 0.0;
    double ss_res = 0.0;

    for (size_t i = 0; i < y.size(); ++i) {
        ss_total += (y[i] - y_mean) * (y[i] - y_mean);
        ss_res += (y[i] - y_pred[i]) * (y[i] - y_pred[i]);
    }

    return 1 - (ss_res / ss_total);
}

void LinearRegression::plot_regression_line(const std::vector<double>& X, const std::vector<double>& y, const std::string& title) const {
    std::vector<double> y_pred = predict(X);
    plt::scatter(X, y);
    plt::plot(X, y_pred);
    plt::title(title);
    plt::xlabel("X");
    plt::ylabel("y");
    plt::show();
}

double LinearRegression::mean(const std::vector<double>& v) const {
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

double LinearRegression::covariance(const std::vector<double>& X, const std::vector<double>& y) const {
    double X_mean = mean(X);
    double y_mean = mean(y);
    double cov = 0.0;

    for (size_t i = 0; i < X.size(); ++i) {
        cov += (X[i] - X_mean) * (y[i] - y_mean);
    }

    return cov / X.size();
}

double LinearRegression::variance(const std::vector<double>& v) const {
    double v_mean = mean(v);
    double var = 0.0;

    for (const auto& val : v) {
        var += (val - v_mean) * (val - v_mean);
    }

    return var / v.size();
}
