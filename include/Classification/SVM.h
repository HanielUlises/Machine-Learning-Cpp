#ifndef SVM_H
#define SVM_H

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

// Enum representing different kernel types for the SVM.
enum class KernelType {
    LINEAR,
    POLYNOMIAL,
    GAUSSIAN
};

// Support Vector Machine (SVM) class definition.
class SVM {
private:
    // Storage for support vectors and related parameters.
    Eigen::MatrixXd support_vectors;
    Eigen::VectorXd alphas;
    Eigen::VectorXd weights;
    double bias;
    double regularization_param; // Regularization parameter (C)
    double tolerance; // Tolerance for stopping criterion
    double epsilon; // Epsilon for support vector threshold
    int max_iterations; // Maximum number of iterations allowed
    KernelType kernel_type; // Type of kernel used for transformation
    int degree; // Degree for polynomial kernel
    double gamma; // Gamma for Gaussian kernel
    double coef0; // Coefficient for polynomial kernel

    // Calculates the kernel function between two feature vectors.
    double kernel_function(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const noexcept;

    // Computes the decision function for a given feature vector.
    double decision_function(const Eigen::VectorXd& x) const noexcept;

    // Optimizes the SVM model based on the provided training data.
    void optimize(const Eigen::MatrixXd& X, const Eigen::VectorXi& y) noexcept;

public:
    // Constructor for initializing the SVM with parameters.
    SVM(double regularization_param, KernelType kernel_type, 
        int max_iterations = 1000, double tolerance = 1e-3, 
        double epsilon = 1e-5, int degree = 3, 
        double gamma = 0.1, double coef0 = 0);

    // Fits the SVM model to the provided training data.
    void fit(const Eigen::MatrixXd& X, const Eigen::VectorXi& y);

    // Predicts the class label for a single feature vector.
    double predict(const Eigen::VectorXd& x) const;

    // Predicts class labels for multiple feature vectors.
    Eigen::VectorXi predict(const Eigen::MatrixXd& X) const;

    // Retrieves the support vectors from the trained SVM model.
    Eigen::MatrixXd get_support_vectors() const;
};

#endif // SVM_H
