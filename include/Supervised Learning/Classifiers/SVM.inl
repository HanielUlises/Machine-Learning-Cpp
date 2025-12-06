#include "SVM.h"

inline double LinearKernel::operator()(const Vec& a, const Vec& b) const {
    if (a.size() != b.size()) throw std::runtime_error("LinearKernel: size mismatch");
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

inline RBFKernel::RBFKernel(double gamma) : gamma_(gamma) {
    if (gamma <= 0.0) throw std::invalid_argument("RBF gamma must be > 0");
}

inline double RBFKernel::operator()(const Vec& a, const Vec& b) const {
    if (a.size() != b.size()) throw std::runtime_error("RBFKernel: size mismatch");
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return std::exp(-gamma_ * sum);
}

inline PolynomialKernel::PolynomialKernel(int degree, double coef0)
    : degree_(degree), coef0_(coef0) {
    if (degree < 0) throw std::invalid_argument("Polynomial degree must be >= 0");
}

inline double PolynomialKernel::operator()(const Vec& a, const Vec& b) const {
    if (a.size() != b.size()) throw std::runtime_error("PolynomialKernel: size mismatch");
    double dot = std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
    return std::pow(dot + coef0_, degree_);
}

// SVM - Complete working SMO implementation (fixed all type issues)
inline SVM::SVM(std::unique_ptr<Kernel> kernel,
                double C,
                int max_iterations,
                double tolerance,
                double eps)
    : kernel_(std::move(kernel))
    , C_(C)
    , max_iterations_(max_iterations)
    , tolerance_(tolerance)
    , eps_(eps)
    , bias_(0.0)
{
    if (C_ <= 0.0 || tolerance_ <= 0.0 || eps_ <= 0.0)
        throw std::invalid_argument("SVM parameters must be positive");
}

inline double SVM::decision_function(const Vec& x) const {
    double f = -bias_;
    for (size_t i = 0; i < support_vectors_.size(); ++i) {
        if (alphas_[i] > eps_) {
            f += alphas_[i] * weights_[i] * (*kernel_)(support_vectors_[i], x);
        }
    }
    return f;
}

inline int SVM::predict(const Vec& x) const {
    return decision_function(x) >= 0.0 ? 1 : -1;
}

inline std::vector<int> SVM::predict(const Mat& X) const {
    std::vector<int> preds;
    preds.reserve(X.size());
    for (const auto& x : X)
        preds.push_back(predict(x));
    return preds;
}

inline SVM::Mat SVM::get_support_vectors() const {
    return support_vectors_;
}

inline void SVM::fit(const Mat& X, const std::vector<int>& y_in) {
    const size_t n = X.size();
    if (n == 0) throw std::invalid_argument("Empty training set");
    if (y_in.size() != n) throw std::invalid_argument("X and y size mismatch");

    // Convert labels to +1/-1
    weights_.resize(n);
    for (size_t i = 0; i < n; ++i) {
        if (y_in[i] != 1 && y_in[i] != -1)
            throw std::invalid_argument("Labels must be +1 or -1");
        weights_[i] = y_in[i]; 
    }

    alphas_.assign(n, 0.0);
    support_vectors_ = X;

    // Precompute kernel matrix 
    std::vector<std::vector<double>> K(n, std::vector<double>(n));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            K[i][j] = (*kernel_)(X[i], X[j]);

    auto kernel_eval = [&](size_t i, size_t j) -> double { return K[i][j]; };

    std::vector<double> error_cache(n, 0.0);

    auto take_step = [&](size_t i1, size_t i2) -> bool {
        if (i1 == i2) return false;

        double alph1 = alphas_[i1];
        double alph2 = alphas_[i2];
        double y1 = weights_[i1];
        double y2 = weights_[i2];
        double E1 = (alph1 > 0 && alph1 < C_) ? error_cache[i1] : decision_function(X[i1]) - y1;
        double E2 = (alph2 > 0 && alph2 < C_) ? error_cache[i2] : decision_function(X[i2]) - y2;

        double s = y1 * y2;

        double L, H;
        if (y1 == y2) {
            L = std::max(0.0, alph1 + alph2 - C_);
            H = std::min(C_, alph1 + alph2);
        } else {
            L = std::max(0.0, alph2 - alph1);
            H = std::min(C_, C_ + alph2 - alph1);
        }
        if (L >= H - eps_) return false;

        double k11 = kernel_eval(i1, i1);
        double k12 = kernel_eval(i1, i2);
        double k22 = kernel_eval(i2, i2);
        double eta = k11 + k22 - 2.0 * k12;

        double a2;
        if (eta > eps_) {
            a2 = alph2 + y2 * (E1 - E2) / eta;
            if (a2 < L) a2 = L;
            else if (a2 > H) a2 = H;
        } else {
            // eta ≈ 0 → use bounds directly
            a2 = (E1 - E2) * y2 > 0 ? L : H;
        }

        if (std::abs(a2 - alph2) < eps_ * (a2 + alph2 + eps_))
            return false;

        double a1 = alph1 + s * (alph2 - a2);

        // Clip to bounds
        if (a1 < 0) { a1 = 0; a2 = alph2 + s * alph1; }
        if (a1 > C_) { a1 = C_; a2 = alph2 + s * (alph1 - C_); }

        // Update bias
        double b1 = E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + bias_;
        double b2 = E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22 + bias_;
        double new_b = (std::abs(a1) < eps_ || std::abs(a1 - C_) < eps_) ? b2 :
                       (std::abs(a2) < eps_ || std::abs(a2 - C_) < eps_) ? b1 : 0.5 * (b1 + b2);

        alphas_[i1] = a1;
        alphas_[i2] = a2;
        bias_ = new_b;

        // Update error cache
        for (size_t i = 0; i < n; ++i) {
            if (0 < alphas_[i] && alphas_[i] < C_) {
                error_cache[i] += y1 * (a1 - alph1) * kernel_eval(i, i1) +
                                  y2 * (a2 - alph2) * kernel_eval(i, i2);
            }
        }
        error_cache[i1] = 0.0;
        error_cache[i2] = 0.0;

        return true;
    };

    auto examine_example = [&](size_t i2) -> bool {
        double y2 = weights_[i2];
        double alph2 = alphas_[i2];
        double E2 = (alph2 > 0 && alph2 < C_) ? error_cache[i2] : decision_function(X[i2]) - y2;
        double r2 = E2 * y2;

        if ((r2 < -tolerance_ && alph2 < C_) || (r2 > tolerance_ && alph2 > 0)) {

            // First heuristic: largest |E1 - E2|
            size_t i1 = 0;
            double max_diff = -1;
            for (size_t k = 0; k < n; ++k) {
                if (alphas_[k] > 0 && alphas_[k] < C_) {
                    double diff = std::abs(error_cache[k] - E2);
                    if (diff > max_diff) {
                        max_diff = diff;
                        i1 = k;
                    }
                }
            }
            if (max_diff > 0 && take_step(i1, i2)) return true;

            // Random non-bound
            for (size_t k = 0; k < n; ++k) {
                if (alphas_[k] > 0 && alphas_[k] < C_)
                    if (take_step(k, i2)) return true;
            }

            // Full scan
            for (size_t k = 0; k < n; ++k)
                if (take_step(k, i2)) return true;
        }
        return false;
    };

    // SMO loop
    int num_changed = 0;
    bool examine_all = true;
    for (int iter = 0; iter < max_iterations_ && (num_changed > 0 || examine_all); ++iter) {
        num_changed = 0;
        if (examine_all) {
            for (size_t i = 0; i < n; ++i)
                num_changed += examine_example(i);
        } else {
            for (size_t i = 0; i < n; ++i)
                if (alphas_[i] > 0 && alphas_[i] < C_)
                    num_changed += examine_example(i);
        }
        if (examine_all) examine_all = false;
        else if (num_changed == 0) examine_all = true;
    }

    // Shrinks to support vectors only
    Mat sv_X;
    Vec sv_alpha;
    Vec sv_y;
    for (size_t i = 0; i < n; ++i) {
        if (alphas_[i] > eps_) {
            sv_X.push_back(X[i]);
            sv_alpha.push_back(alphas_[i]);
            sv_y.push_back(weights_[i]);
        }
    }

    support_vectors_ = std::move(sv_X);
    alphas_ = std::move(sv_alpha);
    weights_ = std::move(sv_y);

    // Final bias from average of support vectors on margin
    if (!support_vectors_.empty()) {
        double sum = 0.0;
        int count = 0;
        for (size_t i = 0; i < support_vectors_.size(); ++i) {
            if (alphas_[i] > eps_ && alphas_[i] < C_ - eps_) {
                sum += weights_[i] - decision_function(support_vectors_[i]);
                ++count;
            }
        }
        if (count > 0) bias_ = sum / count;
    }
}