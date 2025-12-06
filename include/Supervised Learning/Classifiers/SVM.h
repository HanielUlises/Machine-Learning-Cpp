#ifndef SVM_H
#define SVM_H

#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <functional>

class Kernel {
public:
    using Vec = std::vector<double>;

    // Compute kernel value between two vectors
    virtual double operator()(const Vec& a, const Vec& b) const = 0;

    virtual ~Kernel() = default;
};

class LinearKernel : public Kernel {
public:
    double operator()(const Vec& a, const Vec& b) const override;
};

class RBFKernel : public Kernel {
public:
    explicit RBFKernel(double gamma);

    double operator()(const Vec& a, const Vec& b) const override;

private:
    double gamma_;
};

class PolynomialKernel : public Kernel {
public:
    PolynomialKernel(int degree, double coef0 = 1.0);

    double operator()(const Vec& a, const Vec& b) const override;

private:
    int degree_;
    double coef0_;
};

class SVM {
public:
    using Vec = std::vector<double>;
    using Mat = std::vector<Vec>;

    SVM(std::unique_ptr<Kernel> kernel,
        double C = 1.0,
        int max_iterations = 1000,
        double tolerance = 1e-3,
        double eps = 1e-5);

    // Fit the model to training data
    void fit(const Mat& X, const std::vector<int>& y);

    // Predict label of a single sample
    int predict(const Vec& x) const;

    // Predict labels for matrix of samples
    std::vector<int> predict(const Mat& X) const;

    // Get support vectors
    Mat get_support_vectors() const;

private:
    double decision_function(const Vec& x) const;

    void optimize(const Mat& X, const std::vector<int>& y);

private:
    std::unique_ptr<Kernel> kernel_;
    double C_;
    int max_iterations_;
    double tolerance_;
    double eps_;

    Mat support_vectors_;
    Vec alphas_;
    Vec weights_;
    double bias_;
};

#endif // SVM_H
