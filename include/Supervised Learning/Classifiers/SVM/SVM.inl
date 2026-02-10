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
      bias_(0.0),
      error_(Eigen::VectorXd::Zero(data_.size()))
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


inline void SVM::fit()
{
    const std::size_t n = data_.size();

    kernel_cache_.precompute();

    constexpr double tol = 1e-3;        // KKT tolerance
    constexpr double eps = 1e-5;        // minimal alpha step
    constexpr std::size_t max_passes = 10; // SMO stopping criterion

    error_ = -labels_;

    std::size_t passes = 0;

    while (passes < max_passes)
    {
        std::size_t num_changed = 0;

        for (std::size_t i = 0; i < n; ++i)
        {
            const double Ei = error_(i);       // cached error
            const double yi = labels_(i);      // label
            const double ai_old = alpha_(i);   // previous alpha_i

            const bool violates_kkt =
                (yi * Ei < -tol && ai_old < C_) ||
                (yi * Ei >  tol && ai_old > 0.0);

            if (!violates_kkt)
                continue;

            std::size_t j = i; 
            double max_delta = 0.0;

            for (std::size_t k = 0; k < n; ++k)
            {
                if (k == i)
                    continue;

                const double delta = std::abs(Ei - error_(k));
                if (delta > max_delta)
                {
                    max_delta = delta;
                    j = k;
                }
            }

            if (j == i)
                continue;

            const double Ej = error_(j);     // cached error_j
            const double yj = labels_(j);    // label_j
            const double aj_old = alpha_(j); // previous alpha_j

            double L, H; // feasible interval

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

            const double eta = 2.0 * kij - kii - kjj; // second derivative

            if (eta >= 0.0)
                continue;

            alpha_(j) -= yj * (Ei - Ej) / eta; // unconstrained update
            alpha_(j) = std::clamp(alpha_(j), L, H); // project to box

            if (std::abs(alpha_(j) - aj_old) < eps)
            {
                alpha_(j) = aj_old;
                continue;
            }

            alpha_(i) += yi * yj * (aj_old - alpha_(j));

            const double b_old = bias_;

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

            for (std::size_t k = 0; k < n; ++k)
            {
                error_(k) +=
                    (alpha_(i) - ai_old) * yi * kernel_cache_(i, k) +
                    (alpha_(j) - aj_old) * yj * kernel_cache_(j, k) +
                    (bias_ - b_old);
            }

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
