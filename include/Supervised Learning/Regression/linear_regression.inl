#include "linear_regression.h"
#include <stdexcept>
#include <numeric>
#include <cmath>
#include <algorithm>

template<typename T>
inline LinearRegression<T>::LinearRegression(bool fit_intercept, T regularization_param)
    : intercept(0.0), fit_intercept(fit_intercept), regularization_param(regularization_param),
      learning_rate(static_cast<T>(0.01)), max_iterations(1000) {}

template<typename T>
inline void LinearRegression<T>::fit(const std::vector<std::vector<T>>& input_features,
                                     const std::vector<T>& target_variable) {
    if (input_features.size() != target_variable.size()) {
        throw std::invalid_argument("The size of input_features and target_variable must be equal.");
    }

    size_t n_samples = input_features.size();
    size_t n_features = input_features[0].size();

    std::vector<std::vector<T>> X = normalize(input_features);

    if (fit_intercept) {
        intercept = mean(target_variable);
    }

    coefficients.assign(n_features, static_cast<T>(0.0));

    for (size_t iter = 0; iter < max_iterations; ++iter) {
        std::vector<T> preds = predict(input_features);
        std::vector<T> gradient(n_features, static_cast<T>(0.0));
        T intercept_grad = static_cast<T>(0.0);

        for (size_t i = 0; i < n_samples; ++i) {
            T error = preds[i] - target_variable[i];
            if (fit_intercept) {
                intercept_grad += error;
            }
            for (size_t j = 0; j < n_features; ++j) {
                gradient[j] += error * X[i][j];
            }
        }

        if (fit_intercept) {
            intercept -= learning_rate * (intercept_grad / n_samples);
        }

        for (size_t j = 0; j < n_features; ++j) {
            T grad = gradient[j] / n_samples;
            grad += regularization_param * coefficients[j];
            coefficients[j] -= learning_rate * grad;
        }
    }
}

template<typename T>
inline std::vector<T> LinearRegression<T>::predict(const std::vector<std::vector<T>>& input_features) const {
    size_t n_samples = input_features.size();
    size_t n_features = input_features[0].size();

    std::vector<std::vector<T>> X = normalize(input_features);

    std::vector<T> predictions(n_samples, static_cast<T>(0.0));

    for (size_t i = 0; i < n_samples; ++i) {
        T y_hat = fit_intercept ? intercept : static_cast<T>(0.0);
        for (size_t j = 0; j < n_features; ++j) {
            y_hat += coefficients[j] * X[i][j];
        }
        predictions[i] = y_hat;
    }

    return predictions;
}

template<typename T>
inline T LinearRegression<T>::score(const std::vector<std::vector<T>>& input_features,
                                    const std::vector<T>& target_variable) const {
    if (input_features.size() != target_variable.size()) {
        throw std::invalid_argument("The size of input_features and target_variable must be equal.");
    }

    std::vector<T> preds = predict(input_features);
    T mu = mean(target_variable);

    T ss_res = static_cast<T>(0.0);
    T ss_tot = static_cast<T>(0.0);

    for (size_t i = 0; i < target_variable.size(); ++i) {
        T diff = target_variable[i] - preds[i];
        ss_res += diff * diff;
        T d2 = target_variable[i] - mu;
        ss_tot += d2 * d2;
    }

    return static_cast<T>(1.0) - ss_res / ss_tot;
}

template<typename T>
inline T LinearRegression<T>::mean(const std::vector<T>& values) const {
    return std::accumulate(values.begin(), values.end(), static_cast<T>(0.0)) / values.size();
}

template<typename T>
inline T LinearRegression<T>::variance(const std::vector<T>& values) const {
    T mu = mean(values);
    T v = static_cast<T>(0.0);
    for (const auto& x : values) {
        T d = x - mu;
        v += d * d;
    }
    return v / values.size();
}

template<typename T>
inline T LinearRegression<T>::covariance(const std::vector<T>& x, const std::vector<T>& y) const {
    T mx = mean(x);
    T my = mean(y);
    T c = static_cast<T>(0.0);
    for (size_t i = 0; i < x.size(); ++i) {
        c += (x[i] - mx) * (y[i] - my);
    }
    return c / x.size();
}

template<typename T>
inline std::vector<T> LinearRegression<T>::compute_gradient(const std::vector<std::vector<T>>&,
                                                            const std::vector<T>&) const {
    return std::vector<T>(coefficients.size(), static_cast<T>(0.0));
}

template<typename T>
inline std::vector<std::vector<T>> LinearRegression<T>::add_bias_term(const std::vector<std::vector<T>>& input_features) const {
    std::vector<std::vector<T>> out(input_features.size(), std::vector<T>(input_features[0].size() + 1, static_cast<T>(1.0)));
    for (size_t i = 0; i < input_features.size(); ++i) {
        for (size_t j = 0; j < input_features[i].size(); ++j) {
            out[i][j + 1] = input_features[i][j];
        }
    }
    return out;
}

template<typename T>
inline std::vector<std::vector<T>> LinearRegression<T>::normalize(const std::vector<std::vector<T>>& X) const {
    size_t n_samples = X.size();
    size_t n_features = X[0].size();

    std::vector<std::vector<T>> Xn = X;

    for (size_t j = 0; j < n_features; ++j) {
        T mu = static_cast<T>(0.0);
        for (size_t i = 0; i < n_samples; ++i) {
            mu += X[i][j];
        }
        mu /= n_samples;

        T s = static_cast<T>(0.0);
        for (size_t i = 0; i < n_samples; ++i) {
            T d = X[i][j] - mu;
            s += d * d;
        }
        s = std::sqrt(s / n_samples);
        if (s == static_cast<T>(0.0)) s = static_cast<T>(1.0);

        for (size_t i = 0; i < n_samples; ++i) {
            Xn[i][j] = (X[i][j] - mu) / s;
        }
    }

    return Xn;
}

template<typename T>
inline std::vector<T> LinearRegression<T>::get_coefficients() const {
    return coefficients;
}

template<typename T>
inline T LinearRegression<T>::get_intercept() const {
    return intercept;
}
