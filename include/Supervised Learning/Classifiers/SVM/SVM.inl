#pragma once

#include "SVM.hpp"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <limits>

namespace mlpp::classifiers::svm {

template<typename KernelType>
void SVM<KernelType>::init_kernel_cache() const
{
    const Index n = this->alphas_.size();
    kernel_cache_ = std::vector<std::vector<Scalar>>(static_cast<std::size_t>(n),
                    std::vector<Scalar>(static_cast<std::size_t>(n)));

    for (Index i = 0; i < n; ++i) {
        for (Index j = 0; j < n; ++j) {
            kernel_cache_[i][j] = kernel_.eval(X_train_->row(i), X_train_->row(j));
        }
    }
}

template<typename KernelType>
Scalar SVM<KernelType>::get_from_cache(Index i, Index j) const
{
    if constexpr (is_linear_kernel) {
        return X_train_->row(i).dot(X_train_->row(j));
    } else {
        return kernel_cache_[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
    }
}


template<typename KernelType>
Scalar SVM<KernelType>::kernel(Index i, Index j) const
{
    if constexpr (is_linear_kernel) {
        return kernel_.eval(X_train_->row(i), X_train_->row(j));
    } else {
        return kernel_.eval(this->X_sv_.row(i), this->X_sv_.row(j));
    }
}


template<typename KernelType>
void SVM<KernelType>::fit(const Matrix& X, const Vector& y)
{
    const Index n = X.rows();
    X_train_ = &X;

    this->alphas_ = Vector::Zero(n);
    grad_ = -y;  // f_i = 0 initially → grad_i = -y_i

    if constexpr (!is_linear_kernel) {
        init_kernel_cache();
    }

    auto get_K = [&](Index i, Index j) -> Scalar {
        return get_from_cache(i, j);
    };

    Vector y_full = y;

    int num_changed = 0;
    bool examine_all = true;
    int iter = 0;

    while ((examine_all || num_changed > 0) && iter < static_cast<int>(params_.max_iter)) {
        ++iter;
        num_changed = 0;

        if (examine_all) {
            for (Index i = 0; i < n; ++i) {
                // KKT violation check
                const Scalar G_i = grad_(i) + y_full(i);  // actual f(x_i)
                const Scalar yG_i = y_full(i) * G_i;

                if ((this->alphas_(i) < params_.C - params_.eps && yG_i < -params_.tol) ||
                    (this->alphas_(i) > params_.eps && yG_i > params_.tol)) {

                    // Second-order heuristic: find j that maximizes |E_i - E_j|
                    Scalar max_diff = 0;
                    Index j_best = -1;
                    const Scalar E_i = G_i - y_full(i);

                    for (Index j = 0; j < n; ++j) {
                        if (this->alphas_(j) > params_.eps && this->alphas_(j) < params_.C - params_.eps) {
                            const Scalar E_j = grad_(j);
                            const Scalar diff = std::abs(E_i - E_j);
                            if (diff > max_diff) {
                                max_diff = diff;
                                j_best = j;
                            }
                        }
                    }
                    if (j_best == -1) continue;

                    // Take step
                    Scalar old_ai = this->alphas_(i);
                    Scalar old_aj = this->alphas_(j_best);

                    Scalar L, H;
                    if (y_full(i) == y_full(j_best)) {
                        L = std::max(Scalar(0), old_ai + old_aj - params_.C);
                        H = std::min(params_.C, old_ai + old_aj);
                    } else {
                        L = std::max(Scalar(0), old_aj - old_ai);
                        H = std::min(params_.C, params_.C + old_aj - old_ai);
                    }
                    if (L >= H - params_.eps) continue;

                    const Scalar kii = get_K(i, i);
                    const Scalar kjj = get_K(j_best, j_best);
                    const Scalar kij = get_K(i, j_best);
                    const Scalar eta = kii + kjj - 2 * kij;
                    if (eta <= 0) continue;

                    Scalar a_j_new = old_aj + y_full(j_best) * (E_i - (grad_(j_best))) / eta;
                    if (a_j_new > H) a_j_new = H;
                    if (a_j_new < L) a_j_new = L;

                    if (std::abs(a_j_new - old_aj) < params_.eps) continue;

                    Scalar a_i_new = old_ai + y_full(i) * y_full(j_best) * (old_aj - a_j_new);

                    // Update alphas
                    this->alphas_(i) = a_i_new;
                    this->alphas_(j_best) = a_j_new;

                    // Update gradient
                    const Scalar delta_ai = a_i_new - old_ai;
                    const Scalar delta_aj = a_j_new - old_aj;
                    for (Index k = 0; k < n; ++k) {
                        grad_(k) += y_full(i) * delta_ai * get_K(k, i) +
                                    y_full(j_best) * delta_aj * get_K(k, j_best);
                    }

                    ++num_changed;
                }
            }
        } else {

            examine_all = true;
        }

        if (examine_all) examine_all = false;
        else if (num_changed == 0) examine_all = true;
    }

    // Extracts support vectors
    this->sv_indices_.clear();
    for (Index i = 0; i < n; ++i) {
        if (this->alphas_(i) > params_.eps) {
            this->sv_indices_.push_back(i);
        }
    }

    this->X_sv_ = Matrix(this->sv_indices_.size(), X.cols());
    this->y_sv_ = Vector(this->sv_indices_.size());
    for (Index k = 0; k < static_cast<Index>(this->sv_indices_.size()); ++k) {
        Index i = this->sv_indices_[k];
        this->X_sv_.row(k) = X.row(i);
        this->y_sv_(k) = y(i);
    }

    // Compute rho (bias)
    Scalar sum = 0;
    int count = 0;
    for (Index k = 0; k < static_cast<Index>(this->sv_indices_.size()); ++k) {
        Index i = this->sv_indices_[k];
        if (this->alphas_(i) > params_.eps && this->alphas_(i) < params_.C - params_.eps) {
            Scalar f = -y(i);  // subtract y_i
            for (Index m = 0; m < n; ++m) {
                f += this->alphas_(m) * y(m) * get_K(i, m);
            }
            sum += f;
            ++count;
        }
    }
    this->rho_ = (count > 0) ? sum / count : 0;

    std::cout << "[SVM] Training complete. SVs: " << this->sv_indices_.size()
              << ", iterations: ~" << iter << "\n";
}

template<typename KernelType>
Scalar SVM<KernelType>::decision_function(const Vector& x) const
{
    Scalar sum = Scalar(0);
    for (Index k = 0; k < static_cast<Index>(this->sv_indices_.size()); ++k) {
        const Index i = this->sv_indices_[k];
        sum += this->alphas_(i) * this->y_sv_(k) * kernel_.eval(X_train_->row(i), x);
    }
    return sum - this->rho_;
}

} // namespace mlpp::classifiers::svm