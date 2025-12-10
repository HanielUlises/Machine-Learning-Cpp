#pragma once

#include "polynomial_regression.hpp"
#include <stdexcept>
#include <cmath>

template <typename T>
PolynomialFeatures<T>::PolynomialFeatures(unsigned degree, bool include_bias, bool include_interactions)
    : degree_(degree), include_bias_(include_bias), include_interactions_(include_interactions) {
    if (degree_ < 1) degree_ = 1;
}

template <typename T>
std::vector<std::vector<T>> PolynomialFeatures<T>::transform(const std::vector<std::vector<T>>& X) const {
    if (X.empty()) return {};
    size_t n = X.size();
    size_t d = X[0].size();

    std::vector<std::vector<unsigned>> exponents;

    if (include_interactions_) {
        generate_exponent_vectors(d, degree_, exponents);
    } else {
        // per-feature powers: for each feature j include x_j^1 .. x_j^degree (plus bias optionally)
        exponents.reserve((include_bias_ ? 1 : 0) + d * degree_);
        if (include_bias_) exponents.push_back(std::vector<unsigned>(d, 0));
        for (size_t j = 0; j < d; ++j) {
            for (unsigned p = 1; p <= degree_; ++p) {
                std::vector<unsigned> vec(d, 0);
                vec[j] = p;
                exponents.push_back(std::move(vec));
            }
        }
    }

    size_t out_dim = exponents.size();
    std::vector<std::vector<T>> out(n, std::vector<T>(out_dim, static_cast<T>(0)));

    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < out_dim; ++k) {
            T val = static_cast<T>(1);
            const auto& exp = exponents[k];
            for (size_t j = 0; j < d; ++j) {
                unsigned e = exp[j];
                if (e == 0) continue;
                T base = X[i][j];
                T term = static_cast<T>(1);
                for (unsigned q = 0; q < e; ++q) term *= base;
                val *= term;
            }
            out[i][k] = val;
        }
    }

    return out;
}

template <typename T>
size_t PolynomialFeatures<T>::output_dim(size_t n_features) const {
    if (n_features == 0) return 0;
    if (!include_interactions_) {
        return (include_bias_ ? 1u : 0u) + n_features * degree_;
    }
    // number of monomials with degree <= D for F features = C(F + D, D)
    unsigned F = static_cast<unsigned>(n_features);
    unsigned D = degree_;
    unsigned numer = 1;
    unsigned denom = 1;
    unsigned k = D;
    unsigned n = F + D;
    if (k > n - k) k = n - k;
    for (unsigned i = 1; i <= k; ++i) {
        numer *= (n - (k - i));
        denom *= i;
    }
    return numer / denom;
}

template <typename T>
void PolynomialFeatures<T>::generate_exponent_vectors(size_t n_features, unsigned deg,
                                                      std::vector<std::vector<unsigned>>& out) const {
    out.clear();
    if (include_bias_) out.push_back(std::vector<unsigned>(n_features, 0));
    for (unsigned total = 1; total <= deg; ++total) {
        std::vector<unsigned> cur(n_features, 0);
        recurse_exponents(0, n_features, total, cur, out);
    }
}

template <typename T>
void PolynomialFeatures<T>::recurse_exponents(size_t pos, size_t n_features, unsigned remaining,
                                              std::vector<unsigned>& current, std::vector<std::vector<unsigned>>& out) const {
    if (pos + 1 == n_features) {
        current[pos] = remaining;
        out.push_back(current);
        return;
    }
    for (unsigned v = 0; v <= remaining; ++v) {
        current[pos] = v;
        recurse_exponents(pos + 1, n_features, remaining - v, current, out);
    }
}

template <typename T, template<typename> class BaseRegressor>
PolynomialRegression<T, BaseRegressor>::PolynomialRegression(unsigned degree, bool include_bias, bool include_interactions)
    : features_(degree, include_bias, include_interactions), reg_() {}

template <typename T, template<typename> class BaseRegressor>
void PolynomialRegression<T, BaseRegressor>::fit(const std::vector<std::vector<T>>& X, const std::vector<T>& y) {
    if (X.empty() || X.size() != y.size()) throw std::invalid_argument("Invalid data for fit.");
    auto Xp = features_.transform(X);
    reg_.fit(Xp, y);
}

template <typename T, template<typename> class BaseRegressor>
std::vector<T> PolynomialRegression<T, BaseRegressor>::predict(const std::vector<std::vector<T>>& X) const {
    auto Xp = features_.transform(X);
    return reg_.predict(Xp);
}

template <typename T, template<typename> class BaseRegressor>
const std::vector<T>& PolynomialRegression<T, BaseRegressor>::weights() const {
    return reg_.weights();
}

template <typename T, template<typename> class BaseRegressor>
void PolynomialRegression<T, BaseRegressor>::set_regressor(const BaseRegressor<T>& reg) {
    reg_ = reg;
}

template <typename T, template<typename> class BaseRegressor>
BaseRegressor<T>& PolynomialRegression<T, BaseRegressor>::regressor() {
    return reg_;
}
