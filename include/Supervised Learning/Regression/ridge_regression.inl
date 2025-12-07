#pragma once

#include "ridge_regression.h"

template <typename T>
inline RidgeRegression<T>::RidgeRegression(T lambda)
    : lambda_(lambda) {}

template <typename T>
inline void RidgeRegression<T>::fit(const std::vector<std::vector<T>>& X,
                                    const std::vector<T>& y) {
    if (X.empty() || y.empty() || X.size() != y.size())
        throw std::invalid_argument("Invalid dataset size.");

    size_t n = X.size();
    size_t d = X[0].size();

    auto Xt = transpose(X);

    auto XtX = matmul(Xt, X);

    for (size_t i = 0; i < d; ++i)
        XtX[i][i] += lambda_;

    auto XtX_inv = inverse(XtX);

    auto Xt_y = matvec(Xt, y);

    w_ = matvec(XtX_inv, Xt_y);
}

template <typename T>
inline std::vector<T> RidgeRegression<T>::predict(
        const std::vector<std::vector<T>>& X) const {
    if (w_.empty())
        throw std::runtime_error("Model not fitted.");

    std::vector<T> out(X.size(), static_cast<T>(0));
    for (size_t i = 0; i < X.size(); ++i) {
        T s = 0;
        for (size_t j = 0; j < w_.size(); ++j)
            s += X[i][j] * w_[j];
        out[i] = s;
    }
    return out;
}

template <typename T>
inline const std::vector<T>& RidgeRegression<T>::weights() const {
    return w_;
}

template <typename T>
inline void RidgeRegression<T>::set_lambda(T lambda) {
    lambda_ = lambda;
}

template <typename T>
inline T RidgeRegression<T>::get_lambda() const {
    return lambda_;
}

template <typename T>
inline std::vector<std::vector<T>>
RidgeRegression<T>::transpose(const std::vector<std::vector<T>>& A) {
    size_t n = A.size();
    size_t m = A[0].size();
    std::vector<std::vector<T>> B(m, std::vector<T>(n));

    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < m; ++j)
            B[j][i] = A[i][j];

    return B;
}

template <typename T>
inline std::vector<std::vector<T>>
RidgeRegression<T>::matmul(const std::vector<std::vector<T>>& A,
                           const std::vector<std::vector<T>>& B) {
    size_t n = A.size();
    size_t m = A[0].size();
    size_t p = B[0].size();

    std::vector<std::vector<T>> C(n, std::vector<T>(p, static_cast<T>(0)));

    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < p; ++j)
            for (size_t k = 0; k < m; ++k)
                C[i][j] += A[i][k] * B[k][j];

    return C;
}

template <typename T>
inline std::vector<T>
RidgeRegression<T>::matvec(const std::vector<std::vector<T>>& A,
                           const std::vector<T>& x) {
    size_t n = A.size();
    size_t m = x.size();

    std::vector<T> y(n, static_cast<T>(0));

    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < m; ++j)
            y[i] += A[i][j] * x[j];

    return y;
}

template <typename T>
inline std::vector<std::vector<T>>
RidgeRegression<T>::identity(size_t n) {
    std::vector<std::vector<T>> I(n, std::vector<T>(n, static_cast<T>(0)));
    for (size_t i = 0; i < n; ++i)
        I[i][i] = static_cast<T>(1);
    return I;
}

template <typename T>
inline std::vector<std::vector<T>>
RidgeRegression<T>::inverse(std::vector<std::vector<T>> A) {
    size_t n = A.size();
    auto I = identity(n);

    for (size_t i = 0; i < n; ++i) {
        T pivot = A[i][i];
        if (std::fabs(pivot) < static_cast<T>(1e-12))
            throw std::runtime_error("Matrix is singular.");

        T inv_pivot = static_cast<T>(1) / pivot;

        for (size_t j = 0; j < n; ++j) {
            A[i][j] *= inv_pivot;
            I[i][j] *= inv_pivot;
        }

        for (size_t k = 0; k < n; ++k) {
            if (k == i) continue;
            T factor = A[k][i];
            for (size_t j = 0; j < n; ++j) {
                A[k][j] -= factor * A[i][j];
                I[k][j] -= factor * I[i][j];
            }
        }
    }

    return I;
}
