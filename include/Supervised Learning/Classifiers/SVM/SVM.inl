#pragma once

#include <algorithm>
#include <random>
#include <cmath>
#include <limits>

#include "SVM.hpp"

namespace mlpp::classifiers::kernel
{


inline
SVM::SVM(const std::vector<Vector>& data,
         LabelVector labels,
         KernelFunction kernel,
         double C)
    : data_(data),
      labels_(std::move(labels)),
      C_(C),
      kernel_cache_(data_, std::move(kernel)),
      alpha_(AlphaVector::Zero(data_.size())),
      bias_(0.0)
{
}

inline
double SVM::decision(const Vector& x) const
{
    double value = bias_;

    const std::size_t n = data_.size();

    for (std::size_t i = 0; i < n; ++i)
    {
        if (alpha_(i) > 0.0)
        {
            value += alpha_(i) * labels_(i)
                   * kernel_cache_.kernel()(data_[i], x);
        }
    }

    return value;
}

inline
int SVM::predict(const Vector& x) const
{
    return decision(x) >= 0.0 ? +1 : -1;
}

inline
void SVM::compute_bias()
{
    const std::size_t n = data_.size();

    double sum = 0.0;
    std::size_t count = 0;

    for (std::size_t i = 0; i < n; ++i)
    {
        if (alpha_(i) > 0.0 && alpha_(i) < C_)
        {
            double s = 0.0;

            for (std::size_t j = 0; j < n; ++j)
            {
                if (alpha_(j) > 0.0)
                {
                    s += alpha_(j) * labels_(j)
                       * kernel_cache_(j, i);
                }
            }

            sum += labels_(i) - s;
            ++count;
        }
    }

    if (count > 0)
        bias_ = sum / static_cast<double>(count);
}


inline
void SVM::fit()
{
    const std::size_t n = data_.size();

    kernel_cache_.precompute();

    constexpr double tol = 1e-3;
    constexpr double eps = 1e-5;
    constexpr std::size_t max_passes = 10;

    auto error = [&](std::size_t i)
    {
        double s = bias_;

        for (std::size_t k = 0; k < n; ++k)
        {
            if (alpha_(k) > 0.0)
            {
                s += alpha_(k) * labels_(k)
                   * kernel_cache_(k, i);
            }
        }

        return s - labels_(i);
    };

    std::mt19937 rng(1337);
    std::uniform_int_distribution<std::size_t> dist(0, n - 1);

    std::size_t passes = 0;

    while (passes < max_passes)
    {
        std::size_t num_changed = 0;

        for (std::size_t i = 0; i < n; ++i)
        {
            const double Ei = error(i);
            const double yi = labels_(i);
            const double ai = alpha_(i);

            const bool violates_kkt =
                (yi * Ei < -tol && ai < C_) ||
                (yi * Ei >  tol && ai > 0.0);

            if (!violates_kkt)
                continue;

            std::size_t j;
            do { j = dist(rng); } while (j == i);

            const double Ej = error(j);
            const double yj = labels_(j);

            double ai_old = alpha_(i);
            double aj_old = alpha_(j);

            double L, H;

            if (yi != yj)
            {
                L = std::max(0.0, aj_old - ai_old);
                H = std::min(C_, C_ + aj_old - ai_old);
            }
            else
            {
                L = std::max(0.0, ai_old + aj_old - C_);
                H = std::min(C_, ai_old + aj_old);
            }

            if (L >= H)
                continue;

            const double kii = kernel_cache_(i, i);
            const double kjj = kernel_cache_(j, j);
            const double kij = kernel_cache_(i, j);

            const double eta = 2.0 * kij - kii - kjj;

            if (eta >= 0.0)
                continue;

            alpha_(j) -= yj * (Ei - Ej) / eta;
            alpha_(j) = std::clamp(alpha_(j), L, H);

            if (std::abs(alpha_(j) - aj_old) < eps)
            {
                alpha_(j) = aj_old;
                continue;
            }

            alpha_(i) += yi * yj * (aj_old - alpha_(j));

            const double b1 =
                bias_ - Ei
                - yi * (alpha_(i) - ai_old) * kii
                - yj * (alpha_(j) - aj_old) * kij;

            const double b2 =
                bias_ - Ej
                - yi * (alpha_(i) - ai_old) * kij
                - yj * (alpha_(j) - aj_old) * kjj;

            if (alpha_(i) > 0.0 && alpha_(i) < C_)
                bias_ = b1;
            else if (alpha_(j) > 0.0 && alpha_(j) < C_)
                bias_ = b2;
            else
                bias_ = 0.5 * (b1 + b2);

            ++num_changed;
        }

        if (num_changed == 0)
            ++passes;
        else
            passes = 0;
    }

    compute_bias();
}

} // namespace mlpp::classifiers::kernel
