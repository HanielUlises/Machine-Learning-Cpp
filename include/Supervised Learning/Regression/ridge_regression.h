#pragma once

#include <vector>
#include <stdexcept>
#include <cmath>

template <typename T = double>
class RidgeRegression {
public:
    explicit RidgeRegression(T lambda = static_cast<T>(1));

    void fit(const std::vector<std::vector<T>>& X,
             const std::vector<T>& y);

    std::vector<T> predict(const std::vector<std::vector<T>>& X) const;

    const std::vector<T>& weights() const;

    void set_lambda(T lambda);
    T get_lambda() const;

private:
    T lambda_;
    std::vector<T> w_;

    static std::vector<std::vector<T>> transpose(const std::vector<std::vector<T>>& A);

    static std::vector<std::vector<T>> matmul(const std::vector<std::vector<T>>& A,
                                              const std::vector<std::vector<T>>& B);

    static std::vector<T> matvec(const std::vector<std::vector<T>>& A,
                                 const std::vector<T>& x);

    static std::vector<std::vector<T>> identity(size_t n);

    static std::vector<std::vector<T>> inverse(std::vector<std::vector<T>> A);
};

#include "ridge_regression.inl"