#ifndef SVM_H
#define SVM_H

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

enum class KernelType {
    LINEAR,
    POLYNOMIAL,
    GAUSSIAN
};

class SVM {
private:
    Eigen::MatrixXd supportVectors;
    Eigen::VectorXd alphas;
    Eigen::VectorXd weights;
    double bias;
    double C; // Regularization parameter
    double tol; // Tolerance for stopping criterion
    double epsilon; // Epsilon for support vector threshold
    int maxIterations; // Maximum number of iterations
    KernelType kernelType;
    int degree; // Degree for polynomial kernel
    double gamma; // Gamma for Gaussian kernel
    double coef0; // Coefficient for polynomial kernel

    double kernelFunction(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const;
    double decisionFunction(const Eigen::VectorXd& x) const;
    void optimize(const Eigen::MatrixXd& X, const Eigen::VectorXi& y);

public:
    SVM(double C, KernelType kernelType, int maxIterations = 1000, double tol = 1e-3, double epsilon = 1e-5, int degree = 3, double gamma = 0.1, double coef0 = 0);

    // Fit the model
    void fit(const Eigen::MatrixXd& X, const Eigen::VectorXi& y);

    // Predict the class label
    double predict(const Eigen::VectorXd& x) const;

    // Predict multiple samples
    Eigen::VectorXi predict(const Eigen::MatrixXd& X) const;

    // Get support vectors
    Eigen::MatrixXd getSupportVectors() const;
};

#endif // SVM_H
