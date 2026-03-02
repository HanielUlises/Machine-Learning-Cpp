#pragma once

#include <vector>
#include <cstddef>
#include <concepts>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <type_traits>

namespace mlpp::model_validation {
    
template<typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

template<Arithmetic T = std::size_t, typename Label = std::size_t>
class ConfusionMatrix {
public:
    using value_type = T;

    explicit ConfusionMatrix(std::size_t num_classes)
        : K_(num_classes),
          M_(num_classes, std::vector<T>(num_classes, T{}))
    {
        if (K_ == 0)
            throw std::invalid_argument("ConfusionMatrix: num_classes must be > 0.");
    }

    // Compile-time ctor
    template<std::size_t N>
    static constexpr ConfusionMatrix fixed() {
        return ConfusionMatrix(N);
    }

    // Update one sample
    void update(const Label& y_true, const Label& y_pred) noexcept {
        const std::size_t t = static_cast<std::size_t>(y_true);
        const std::size_t p = static_cast<std::size_t>(y_pred);
        if (t < K_ && p < K_)
            M_[t][p] += T(1);
    }

    void clear() noexcept {
        for (auto& row : M_)
            std::fill(row.begin(), row.end(), T{});
    }

    [[nodiscard]] std::size_t num_classes() const noexcept { return K_; }

    [[nodiscard]] const std::vector<std::vector<T>>& data() const noexcept {
        return M_;
    }

    [[nodiscard]] const std::vector<T>& operator[](std::size_t i) const noexcept {
        return M_[i];
    }

    [[nodiscard]] std::vector<T>& operator[](std::size_t i) noexcept {
        return M_[i];
    }

    [[nodiscard]] T trace() const noexcept {
        T s{};
        for (std::size_t i = 0; i < K_; ++i)
            s += M_[i][i];
        return s;
    }

    [[nodiscard]] T total() const noexcept {
        T s{};
        for (const auto& r : M_)
            for (T v : r)
                s += v;
        return s;
    }

    void print(std::ostream& os = std::cout, int width = 8) const {
        os << "Confusion Matrix (" << K_ << " classes):\n";
        for (std::size_t i = 0; i < K_; ++i) {
            for (std::size_t j = 0; j < K_; ++j)
                os << std::setw(width) << M_[i][j];
            os << '\n';
        }
    }

private:
    std::size_t K_;
    std::vector<std::vector<T>> M_;
};

}