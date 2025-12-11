// include/mlpp/supervised_learning/classifiers/svm/svm.h
#pragma once

#include <Eigen/Dense>
#include <vector>
#include <cstddef>
#include <cassert>
#include <cmath>
#include <type_traits>

namespace mlpp::classifiers::svm {

// Parameters
struct Parameters {
    double C           = 1.0;     // Regularization parameter (C-SVM)
    double nu          = 0.5;     // nu parameter (nu-SVM)
    double tol         = 1e-3;    // Tolerance for KKT conditions
    double eps         = 1e-12;   // Numerical zero
    std::size_t max_iter = 100000;
    bool   shrinking   = true;
    bool   probability = false;   // Platt scaling (future)
    bool   use_nu      = false;   // false -> C-SVM, true -> v-SVM
};

// Kernel Functors
namespace kernel {

template<typename Scalar = double>
struct Linear {
    using scalar_t = Scalar;
    static constexpr bool is_linear = true;

    static Scalar eval(auto const& a, auto const& b) {
        return a.dot(b);
    }
};

template<typename Scalar = double>
struct Polynomial {
    using scalar_t = Scalar;
    static constexpr bool is_linear = false;

    Scalar gamma = 1.0;
    Scalar coef0 = 1.0;
    int    degree = 3;

    Polynomial(Scalar g = 1.0, Scalar c = 1.0, int d = 3)
        : gamma(g), coef0(c), degree(d) {}

    Scalar eval(auto const& a, auto const& b) const {
        return std::pow(gamma * a.dot(b) + coef0, degree);
    }
};

template<typename Scalar = double>
struct RBF {
    using scalar_t = Scalar;
    static constexpr bool is_linear = false;

    Scalar gamma;

    explicit RBF(Scalar g) : gamma(g) {}

    Scalar eval(auto const& a, auto const& b) const {
        return std::exp(-gamma * (a - b).squaredNorm());
    }
};

template<typename Scalar = double>
struct Sigmoid {
    using scalar_t = Scalar;
    static constexpr bool is_linear = false;

    Scalar gamma = 1.0;
    Scalar coef0 = -1.0;

    Sigmoid(Scalar g = 1.0, Scalar c = -1.0) : gamma(g), coef0(c) {}

    Scalar eval(auto const& a, auto const& b) const {
        return std::tanh(gamma * a.dot(b) + coef0);
    }
};

} // namespace kernel


template<typename Scalar = double>
class SVMBase {
public:
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Index  = Eigen::Index;

    virtual ~SVMBase() = default;

    virtual void fit(const Matrix& X, const Vector& y) = 0;

    // Decision function f(x) = sum alpha_i y_i K(x_i, x) - rho
    virtual Scalar decision_function(const Vector& x) const = 0;

    Scalar predict(const Vector& x) const {
        return decision_function(x) > Scalar(0) ? Scalar(1) : Scalar(-1);
    }

    const Vector& alphas()          const noexcept { return alphas_; }
    Scalar        bias()            const noexcept { return -rho_; }   // LIBSVM convention
    Scalar        rho()             const noexcept { return rho_; }
    const std::vector<Index>& support_indices() const noexcept { return sv_indices_; }
    const Matrix& support_vectors() const noexcept { return X_sv_; }
    const Vector& support_labels()  const noexcept { return y_sv_; }

protected:
    Vector alphas_;                     // Lagrange multipliers (full size n)
    Scalar rho_ = Scalar(0);            // bias term (LIBSVM: decision = ... - rho)
    Matrix X_sv_;                       // support vectors only
    Vector y_sv_;                       // labels of support vectors
    std::vector<Index> sv_indices_;

    virtual Scalar kernel(Index i, Index j) const = 0;
};

template<typename KernelType>
class SVM final : public SVMBase<typename KernelType::scalar_t> {
public:
    using Scalar   = typename KernelType::scalar_t;
    using Base     = SVMBase<Scalar>;
    using typename Base::Matrix;
    using typename Base::Vector;
    using typename Base::Index;

    static constexpr bool is_linear_kernel =
        std::is_same_v<KernelType, kernel::Linear<Scalar>> || KernelType::is_linear;

    explicit SVM(Parameters params = {}, KernelType kern = KernelType{})
        : params_(std::move(params)), kernel_(std::move(kern)) {}

    void fit(const Matrix& X, const Vector& y) override;
    Scalar decision_function(const Vector& x) const override;

private:
    // Kernel evaluation (delegated to functor)
    Scalar kernel(Index i, Index j) const override {
        if constexpr (is_linear_kernel) {
            return kernel_.eval(X_train_->row(i), X_train_->row(j));
        } else {
            return kernel_.eval(X_sv_.row(i), X_sv_.row(j));
        }
    }

    Parameters   params_;
    KernelType   kernel_;
    const Matrix* X_train_ = nullptr;   // full training data (needed for linear case)

    // SMO internal state
    Vector grad_;                       // ∇L = f(x_i) - y_i  (current error cache)

    // Core SMO routines
    bool select_working_set(Index& i_out, Index& j_out) const;
    bool take_step(Index i, Index j);
    void update_alpha_and_grad(Index i, Scalar delta_alpha_i);
    void shrink();
    void reconstruct_gradient();

    // Kernel cache (only for non-linear kernels)
    mutable std::vector<std::vector<Scalar>> kernel_cache_;
    void init_kernel_cache() const;
    Scalar get_from_cache(Index i, Index j) const;
};

} // namespace mlpp::classifiers::svm

// Implementation
#include "SVM.inl"