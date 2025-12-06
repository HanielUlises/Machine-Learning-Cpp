#ifndef METRICS_H
#define METRICS_H

#include "confusion_matrix.h"
#include <cstddef>
#include <type_traits>

// Computes precision, recall, F1, IoU, macro/micro aggregates.
template<typename CM>
class Metrics {
public:
    using T = typename std::remove_cvref_t<CM>::value_type;

    explicit Metrics(const CM& cm)
        : cm_(cm), K_(cm.num_classes()) {}

    // Basic counts

    [[nodiscard]] T tp(std::size_t k) const noexcept {
        return cm_[k][k];
    }

    [[nodiscard]] T fp(std::size_t k) const noexcept {
        T s{};
        for (std::size_t i = 0; i < K_; ++i)
            s += cm_[i][k];
        return s - cm_[k][k];
    }

    [[nodiscard]] T fn(std::size_t k) const noexcept {
        T s{};
        for (std::size_t j = 0; j < K_; ++j)
            s += cm_[k][j];
        return s - cm_[k][k];
    }

    // Precision / Recall / F1

    [[nodiscard]] double precision(std::size_t k) const noexcept {
        const T d = tp(k) + fp(k);
        return (d == 0) ? 0.0 : double(tp(k)) / d;
    }

    [[nodiscard]] double recall(std::size_t k) const noexcept {
        const T d = tp(k) + fn(k);
        return (d == 0) ? 0.0 : double(tp(k)) / d;
    }

    [[nodiscard]] double f1(std::size_t k) const noexcept {
        const double p = precision(k);
        const double r = recall(k);
        const double s = p + r;
        return (s == 0.0) ? 0.0 : 2.0 * p * r / s;
    }

    [[nodiscard]] double iou(std::size_t k) const noexcept {
        const T denom = tp(k) + fp(k) + fn(k);
        return (denom == 0) ? 0.0 : double(tp(k)) / denom;
    }

    // Macro averages
    [[nodiscard]] double macro_precision() const noexcept {
        double s{};
        for (std::size_t k = 0; k < K_; ++k) s += precision(k);
        return s / K_;
    }

    [[nodiscard]] double macro_recall() const noexcept {
        double s{};
        for (std::size_t k = 0; k < K_; ++k) s += recall(k);
        return s / K_;
    }

    [[nodiscard]] double macro_f1() const noexcept {
        double s{};
        for (std::size_t k = 0; k < K_; ++k) s += f1(k);
        return s / K_;
    }

    [[nodiscard]] double mean_iou() const noexcept {
        double s{};
        for (std::size_t k = 0; k < K_; ++k) s += iou(k);
        return s / K_;
    }

    // Micro averages -------------------------------------------------

    [[nodiscard]] double micro_precision() const noexcept {
        T TP{}, FP{};
        for (std::size_t k = 0; k < K_; ++k) {
            TP += tp(k);
            FP += fp(k);
        }
        return (TP + FP == 0) ? 0.0 : double(TP) / (TP + FP);
    }

    [[nodiscard]] double micro_recall() const noexcept {
        T TP{}, FN{};
        for (std::size_t k = 0; k < K_; ++k) {
            TP += tp(k);
            FN += fn(k);
        }
        return (TP + FN == 0) ? 0.0 : double(TP) / (TP + FN);
    }

    [[nodiscard]] double micro_f1() const noexcept {
        const double p = micro_precision();
        const double r = micro_recall();
        const double s = p + r;
        return (s == 0.0) ? 0.0 : 2.0 * p * r / s;
    }

private:
    const CM& cm_;
    std::size_t K_;
};

#endif // METRICS_H
