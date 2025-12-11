#pragma once

#include <vector>
#include <cmath>
#include <type_traits>
#include <stdexcept>

namespace mlpp::losses {

// ---------------------------
// Concepts
// ---------------------------
template<typename T>
concept Arithmetic = std::is_arithmetic_v<T>;


// ============================================================
// Regression Losses
// ============================================================

// Mean Squared Error (MSE)
template<Arithmetic T>
T mse(const std::vector<T>& y_true, const std::vector<T>& y_pred) {
    if (y_true.size() != y_pred.size())
        throw std::invalid_argument("mse: y_true and y_pred size mismatch.");

    T sum = 0;
    for (std::size_t i = 0; i < y_true.size(); ++i)
        sum += (y_pred[i] - y_true[i]) * (y_pred[i] - y_true[i]);

    return sum / static_cast<T>(y_true.size());
}

// Mean Absolute Error (MAE)
template<Arithmetic T>
T mae(const std::vector<T>& y_true, const std::vector<T>& y_pred) {
    if (y_true.size() != y_pred.size())
        throw std::invalid_argument("mae: y_true and y_pred size mismatch.");

    T sum = 0;
    for (std::size_t i = 0; i < y_true.size(); ++i)
        sum += std::abs(y_pred[i] - y_true[i]);

    return sum / static_cast<T>(y_true.size());
}

// Huber Loss
template<Arithmetic T>
T huber(const std::vector<T>& y_true, const std::vector<T>& y_pred, T delta = static_cast<T>(1)) {
    if (y_true.size() != y_pred.size())
        throw std::invalid_argument("huber: y_true and y_pred size mismatch.");

    T sum = 0;
    for (std::size_t i = 0; i < y_true.size(); ++i) {
        T r = y_pred[i] - y_true[i];
        if (std::abs(r) <= delta)
            sum += static_cast<T>(0.5) * r * r;
        else
            sum += delta * (std::abs(r) - static_cast<T>(0.5) * delta);
    }

    return sum / static_cast<T>(y_true.size());
}


// ============================================================
// Classification Losses
// ============================================================

// Binary Cross Entropy
template<Arithmetic T>
T binary_cross_entropy(const std::vector<T>& y_true,
                       const std::vector<T>& y_pred) {
    if (y_true.size() != y_pred.size())
        throw std::invalid_argument("binary_cross_entropy: size mismatch.");

    const T eps = static_cast<T>(1e-12);
    T sum = 0;

    for (std::size_t i = 0; i < y_true.size(); ++i) {
        T p = std::clamp(y_pred[i], eps, static_cast<T>(1) - eps);
        sum += y_true[i] * std::log(p) + (1 - y_true[i]) * std::log(1 - p);
    }

    return -sum / static_cast<T>(y_true.size());
}


// Multiclass Cross Entropy
template<Arithmetic T>
T multiclass_cross_entropy(const std::vector<std::vector<T>>& y_true,
                           const std::vector<std::vector<T>>& y_pred) {
    if (y_true.size() != y_pred.size())
        throw std::invalid_argument("multiclass_cross_entropy: outer size mismatch.");

    const T eps = static_cast<T>(1e-12);
    T sum = 0;

    for (std::size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i].size() != y_pred[i].size())
            throw std::invalid_argument("multiclass_cross_entropy: inner size mismatch.");

        for (std::size_t k = 0; k < y_true[i].size(); ++k) {
            T p = std::clamp(y_pred[i][k], eps, static_cast<T>(1) - eps);
            sum += y_true[i][k] * std::log(p);
        }
    }

    return -sum / static_cast<T>(y_true.size());
}


// Hinge Loss
template<Arithmetic T>
T hinge_loss(const std::vector<T>& y_true,
             const std::vector<T>& y_pred) {
    if (y_true.size() != y_pred.size())
        throw std::invalid_argument("hinge_loss: size mismatch.");

    T sum = 0;
    for (std::size_t i = 0; i < y_true.size(); ++i)
        sum += std::max(static_cast<T>(0), static_cast<T>(1) - y_true[i] * y_pred[i]);

    return sum / static_cast<T>(y_true.size());
}


// Squared Hinge Loss
template<Arithmetic T>
T squared_hinge_loss(const std::vector<T>& y_true,
                     const std::vector<T>& y_pred) {
    if (y_true.size() != y_pred.size())
        throw std::invalid_argument("squared_hinge_loss: size mismatch.");

    T sum = 0;
    for (std::size_t i = 0; i < y_true.size(); ++i) {
        T margin = static_cast<T>(1) - y_true[i] * y_pred[i];
        sum += std::max(static_cast<T>(0), margin) * std::max(static_cast<T>(0), margin);
    }

    return sum / static_cast<T>(y_true.size());
}


// ============================================================
// Regularization Terms
// ============================================================

template<Arithmetic T>
T l1_penalty(const std::vector<T>& weights) {
    T sum = 0;
    for (auto w : weights) sum += std::abs(w);
    return sum;
}

template<Arithmetic T>
T l2_penalty(const std::vector<T>& weights) {
    T sum = 0;
    for (auto w : weights) sum += w * w;
    return sum;
}

template<Arithmetic T>
T elastic_net_penalty(const std::vector<T>& weights, T alpha, T l1_ratio) {
    return alpha * (l1_ratio * l1_penalty(weights) +
                    (static_cast<T>(1) - l1_ratio) * l2_penalty(weights));
}

} // namespace yourlib::losses
