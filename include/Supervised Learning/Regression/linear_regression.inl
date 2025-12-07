#include "linear_regression.h"
#include <stdexcept>
#include <numeric>
#include <cmath>
#include <algorithm>

template<typename T>
inline LinearRegression<T>::LinearRegression(bool fit_intercept, T regularization_param)
    : intercept(static_cast<T>(0.0)),
      fit_intercept(fit_intercept),
      regularization_param(regularization_param),
      learning_rate(static_cast<T>(0.01)),
      max_iterations(1000)
{}

template<typename T>
inline void LinearRegression<T>::fit(const std::vector<std::vector<T>>& input_features,
                                     const std::vector<T>& target_variable) {
    if (input_features.empty() || input_features[0].empty()) {
        throw std::invalid_argument("input_features must be non-empty.");
    }
    if (input_features.size() != target_variable.size()) {
        throw std::invalid_argument("The size of input_features and target_variable must be equal.");
    }

    size_t n_samples = input_features.size();
    size_t n_features = input_features[0].size();

    feature_means_.assign(n_features, static_cast<T>(0.0));
    feature_stds_.assign(n_features, static_cast<T>(0.0));

    for (size_t j = 0; j < n_features; ++j) {
        T sum = static_cast<T>(0.0);
        for (size_t i = 0; i < n_samples; ++i) sum += input_features[i][j];
        feature_means_[j] = sum / static_cast<T>(n_samples);
        T var = static_cast<T>(0.0);
        for (size_t i = 0; i < n_samples; ++i) {
            T d = input_features[i][j] - feature_means_[j];
            var += d * d;
        }
        T s = std::sqrt(var / static_cast<T>(n_samples));
        if (s == static_cast<T>(0.0)) s = static_cast<T>(1.0);
        feature_stds_[j] = s;
    }

    std::vector<std::vector<T>> Xn(n_samples, std::vector<T>(n_features));
    for (size_t i = 0; i < n_samples; ++i) {
        for (size_t j = 0; j < n_features; ++j) {
            Xn[i][j] = (input_features[i][j] - feature_means_[j]) / feature_stds_[j];
        }
    }

    coefficients.assign(n_features, static_cast<T>(0.0));
    if (fit_intercept) intercept = static_cast<T>(0.0);

    for (size_t iter = 0; iter < max_iterations; ++iter) {
        std::vector<T> preds(n_samples, static_cast<T>(0.0));
        for (size_t i = 0; i < n_samples; ++i) {
            T p = fit_intercept ? intercept : static_cast<T>(0.0);
            for (size_t j = 0; j < n_features; ++j) p += coefficients[j] * Xn[i][j];
            preds[i] = p;
        }

        std::vector<T> grad(n_features, static_cast<T>(0.0));
        T intercept_grad = static_cast<T>(0.0);

        for (size_t i = 0; i < n_samples; ++i) {
            T error = preds[i] - target_variable[i];
            if (fit_intercept) intercept_grad += error;
            for (size_t j = 0; j < n_features; ++j) grad[j] += error * Xn[i][j];
        }

        if (fit_intercept) {
            intercept -= learning_rate * (intercept_grad / static_cast<T>(n_samples));
        }

        for (size_t j = 0; j < n_features; ++j) {
            T g = grad[j] / static_cast<T>(n_samples);
            g += regularization_param * coefficients[j];
            coefficients[j] -= learning_rate * g;
        }
    }

    fitted_ = true;
}

template<typename T>
inline std::vector<T> LinearRegression<T>::predict(const std::vector<std::vector<T>>& input_features) const {
    if (!fitted_) throw std::runtime_error("Model has not been fitted.");
    if (input_features.empty()) return {};

    size_t n_samples = input_features.size();
    size_t n_features = input_features[0].size();
    if (n_features != feature_means_.size()) {
        throw std::invalid_argument("Number of features in predict does not match training.");
    }

    std::vector<std::vector<T>> Xn(n_samples, std::vector<T>(n_features));
    for (size_t i = 0; i < n_samples; ++i) {
        for (size_t j = 0; j < n_features; ++j) {
            Xn[i][j] = (input_features[i][j] - feature_means_[j]) / feature_stds_[j];
        }
    }

    std::vector<T> preds(n_samples, static_cast<T>(0.0));
    for (size_t i = 0; i < n_samples; ++i) {
        T p = fit_intercept ? intercept : static_cast<T>(0.0);
        for (size_t j = 0; j < n_features; ++j) p += coefficients[j] * Xn[i][j];
        preds[i] = p;
    }
    return preds;
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
        T d = target_variable[i] - preds[i];
        ss_res += d * d;
        T t = target_variable[i] - mu;
        ss_tot += t * t;
    }
    if (ss_tot == static_cast<T>(0.0)) return static_cast<T>(0.0);
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
    if (x.size() != y.size()) throw std::invalid_argument("Vectors must have the same length for covariance.");
    T mx = mean(x);
    T my = mean(y);
    T c = static_cast<T>(0.0);
    for (size_t i = 0; i < x.size(); ++i) c += (x[i] - mx) * (y[i] - my);
    return c / x.size();
}

template<typename T>
inline std::vector<T> LinearRegression<T>::compute_gradient(const std::vector<std::vector<T>>& input_features,
                                                            const std::vector<T>& target_values) const {
    if (!fitted_) throw std::runtime_error("Model has not been fitted; gradient undefined.");
    size_t n_samples = input_features.size();
    size_t n_features = input_features[0].size();
    std::vector<std::vector<T>> Xn(n_samples, std::vector<T>(n_features));
    for (size_t i = 0; i < n_samples; ++i)
        for (size_t j = 0; j < n_features; ++j)
            Xn[i][j] = (input_features[i][j] - feature_means_[j]) / feature_stds_[j];

    std::vector<T> gradient(n_features, static_cast<T>(0.0));
    for (size_t i = 0; i < n_samples; ++i) {
        T pred = fit_intercept ? intercept : static_cast<T>(0.0);
        for (size_t j = 0; j < n_features; ++j) pred += coefficients[j] * Xn[i][j];
        T err = pred - target_values[i];
        for (size_t j = 0; j < n_features; ++j) gradient[j] += err * Xn[i][j];
    }
    for (size_t j = 0; j < n_features; ++j) {
        gradient[j] = gradient[j] / static_cast<T>(n_samples) + regularization_param * coefficients[j];
    }
    return gradient;
}

template<typename T>
inline std::vector<std::vector<T>> LinearRegression<T>::add_bias_term(const std::vector<std::vector<T>>& input_features) const {
    std::vector<std::vector<T>> out(input_features.size(), std::vector<T>(input_features[0].size() + 1, static_cast<T>(1.0)));
    for (size_t i = 0; i < input_features.size(); ++i)
        for (size_t j = 0; j < input_features[i].size(); ++j)
            out[i][j + 1] = input_features[i][j];
    return out;
}

template<typename T>
inline std::vector<std::vector<T>> LinearRegression<T>::normalize(const std::vector<std::vector<T>>& X) const {
    if (feature_means_.empty()) throw std::runtime_error("Normalization statistics are not initialized. Fit the model first.");
    size_t n_samples = X.size();
    size_t n_features = X[0].size();
    if (n_features != feature_means_.size()) throw std::invalid_argument("Feature count mismatch for normalization.");
    std::vector<std::vector<T>> Xn(n_samples, std::vector<T>(n_features));
    for (size_t i = 0; i < n_samples; ++i)
        for (size_t j = 0; j < n_features; ++j)
            Xn[i][j] = (X[i][j] - feature_means_[j]) / feature_stds_[j];
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
