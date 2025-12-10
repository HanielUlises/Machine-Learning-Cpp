#pragma once

#include <vector>
#include <cstddef>

template <typename T = double>
class PolynomialFeatures {
public:
    explicit PolynomialFeatures(unsigned degree = 2, bool include_bias = true, bool include_interactions = false);

    std::vector<std::vector<T>> transform(const std::vector<std::vector<T>>& X) const;

    size_t output_dim(size_t n_features) const;

private:
    unsigned degree_;
    bool include_bias_;
    bool include_interactions_;

    void generate_exponent_vectors(size_t n_features, unsigned deg,
                                   std::vector<std::vector<unsigned>>& out) const;
    void recurse_exponents(size_t pos, size_t n_features, unsigned remaining,
                           std::vector<unsigned>& current, std::vector<std::vector<unsigned>>& out) const;
};

template <typename T = double, template<typename> class BaseRegressor = class /* forward placeholder */ >
class PolynomialRegression {
public:
    explicit PolynomialRegression(unsigned degree = 2, bool include_bias = true, bool include_interactions = false);

    void fit(const std::vector<std::vector<T>>& X, const std::vector<T>& y);

    std::vector<T> predict(const std::vector<std::vector<T>>& X) const;

    const std::vector<T>& weights() const;

    void set_regressor(const BaseRegressor<T>& reg);
    BaseRegressor<T>& regressor();

private:
    PolynomialFeatures<T> features_;
    BaseRegressor<T> reg_;
};

#include "polynomial_regression.inl"