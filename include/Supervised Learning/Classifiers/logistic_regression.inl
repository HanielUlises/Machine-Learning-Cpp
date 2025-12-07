#pragma once

#include "logistic_regression.h"

#include <cassert>

template <typename T>
LogisticRegression<T>::LogisticRegression(T learning_rate,
                                          std::size_t max_iterations,
                                          bool fit_intercept)
    : weights{},
      intercept(static_cast<T>(0)),
      learning_rate(learning_rate),
      max_iterations(max_iterations),
      fit_intercept(fit_intercept) {}


template <typename T>
T LogisticRegression<T>::sigmoid(T z) const
{
    if (z >= 0) {
        T exp_neg = std::exp(-z);
        return static_cast<T>(1) / (static_cast<T>(1) + exp_neg);
    } else {
        T exp_pos = std::exp(z);
        return exp_pos / (static_cast<T>(1) + exp_pos);
    }
}


template <typename T>
T LogisticRegression<T>::compute_logit(const std::vector<T>& x) const
{
    assert(x.size() == weights.size());

    T z = intercept;
    for (std::size_t i = 0; i < x.size(); ++i) {
        z += weights[i] * x[i];
    }
    return z;
}


template <typename T>
std::vector<T> LogisticRegression<T>::compute_gradient(
        const std::vector<std::vector<T>>& input_features,
        const std::vector<T>& target_values) const
{
    std::size_t n_samples = input_features.size();
    std::size_t n_features = weights.size();

    std::vector<T> gradient(n_features, static_cast<T>(0));
    T intercept_grad = static_cast<T>(0);

    for (std::size_t i = 0; i < n_samples; ++i) {
        T prediction = sigmoid(compute_logit(input_features[i]));
        T error = prediction - target_values[i];

        for (std::size_t j = 0; j < n_features; ++j) {
            gradient[j] += error * input_features[i][j];
        }

        if (fit_intercept)
            intercept_grad += error;
    }

    for (std::size_t j = 0; j < n_features; ++j)
        gradient[j] /= n_samples;

    if (fit_intercept)
        intercept_grad /= n_samples;

    gradient.push_back(intercept_grad);

    return gradient;
}


template <typename T>
void LogisticRegression<T>::fit(const std::vector<std::vector<T>>& input_features,
                                const std::vector<T>& target_values)
{
    if (input_features.empty())
        throw std::runtime_error("Input features must not be empty.");

    if (input_features.size() != target_values.size())
        throw std::runtime_error("Feature and target sizes differ.");

    std::size_t n_samples = input_features.size();
    std::size_t n_features = input_features[0].size();

    weights.assign(n_features, static_cast<T>(0));
    intercept = static_cast<T>(0);

    for (std::size_t iter = 0; iter < max_iterations; ++iter) {
        auto gradient = compute_gradient(input_features, target_values);

        for (std::size_t j = 0; j < n_features; ++j)
            weights[j] -= learning_rate * gradient[j];

        if (fit_intercept)
            intercept -= learning_rate * gradient.back();
    }
}


template <typename T>
std::vector<T> LogisticRegression<T>::predict_proba(
        const std::vector<std::vector<T>>& input_features) const
{
    std::vector<T> probabilities;
    probabilities.reserve(input_features.size());

    for (const auto& row : input_features) {
        if (row.size() != weights.size())
            throw std::runtime_error("Input feature size mismatch.");

        probabilities.push_back(sigmoid(compute_logit(row)));
    }

    return probabilities;
}


template <typename T>
std::vector<int> LogisticRegression<T>::predict(
        const std::vector<std::vector<T>>& input_features) const
{
    auto probabilities = predict_proba(input_features);

    std::vector<int> predictions;
    predictions.reserve(probabilities.size());

    for (auto p : probabilities)
        predictions.push_back(p >= static_cast<T>(0.5) ? 1 : 0);

    return predictions;
}
