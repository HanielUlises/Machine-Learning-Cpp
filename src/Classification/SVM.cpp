#include "SVM.h"

// Constructor initializes the SVM parameters and sets the bias to zero.
SVM::SVM(double regularization_param, KernelType kernel_type, int max_iterations, double tolerance, double epsilon, int degree, double gamma, double coef0)
    : regularization_param(regularization_param), kernel_type(kernel_type), max_iterations(max_iterations), tolerance(tolerance),
      epsilon(epsilon), degree(degree), gamma(gamma), coef0(coef0), bias(0.0) {}

// Computes the kernel function based on the specified kernel type.
double SVM::kernel_function(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const {
    double result = 0.0;
    switch (kernel_type) {
        case KernelType::LINEAR:
            result = x1.dot(x2);  // Linear kernel
            break;
        case KernelType::POLYNOMIAL:
            result = std::pow(x1.dot(x2) + coef0, degree);  // Polynomial kernel
            break;
        case KernelType::GAUSSIAN:
            result = std::exp(-gamma * (x1 - x2).squaredNorm());  // Gaussian (RBF) kernel
            break;
    }
    return result;
}

// Evaluates the decision function for the provided feature vector.
double SVM::decision_function(const Eigen::VectorXd& x) const {
    double result = 0.0;
    for (int i = 0; i < support_vectors.rows(); ++i) {
        result += alphas[i] * weights[i] * kernel_function(support_vectors.row(i), x);
    }
    return result + bias;
}

// Optimizes the SVM parameters using the provided training data.
void SVM::optimize(const Eigen::MatrixXd& X, const Eigen::VectorXi& y) {
    int m = X.rows();
    alphas = Eigen::VectorXd::Zero(m);
    weights = Eigen::VectorXd::Zero(m);
    bias = 0.0;

    int iteration = 0;
    while (iteration < max_iterations) {
        int alpha_changed = 0;
        for (int i = 0; i < m; ++i) {
            double Ei = decision_function(X.row(i)) - y[i];
            if ((y[i] * Ei < -tolerance && alphas[i] < regularization_param) || 
                (y[i] * Ei > tolerance && alphas[i] > 0)) {
                // Randomly select j != i
                int j = i;
                while (j == i) {
                    j = rand() % m;
                }

                double Ej = decision_function(X.row(j)) - y[j];
                double alpha_i_old = alphas[i];
                double alpha_j_old = alphas[j];

                // Compute L and H
                double L, H;
                if (y[i] != y[j]) {
                    L = std::max(0.0, alphas[j] - alphas[i]);
                    H = std::min(regularization_param, regularization_param + alphas[j] - alphas[i]);
                } else {
                    L = std::max(0.0, alphas[i] + alphas[j] - regularization_param);
                    H = std::min(regularization_param, alphas[i] + alphas[j]);
                }

                if (L == H) continue;

                double eta = 2.0 * kernel_function(X.row(i), X.row(j)) 
                             - kernel_function(X.row(i), X.row(i)) 
                             - kernel_function(X.row(j), X.row(j));
                if (eta >= 0) continue;

                // Update alpha[j]
                alphas[j] -= y[j] * (Ei - Ej) / eta;

                // Clip alpha[j]
                alphas[j] = std::clamp(alphas[j], L, H);

                if (std::abs(alphas[j] - alpha_j_old) < epsilon) continue;

                // Update alpha[i]
                alphas[i] += y[i] * y[j] * (alpha_j_old - alphas[j]);

                // Update bias
                double b1 = bias - Ei - y[i] * (alphas[i] - alpha_i_old) * kernel_function(X.row(i), X.row(i)) 
                                 - y[j] * (alphas[j] - alpha_j_old) * kernel_function(X.row(i), X.row(j));
                double b2 = bias - Ej - y[i] * (alphas[i] - alpha_i_old) * kernel_function(X.row(i), X.row(j)) 
                                 - y[j] * (alphas[j] - alpha_j_old) * kernel_function(X.row(j), X.row(j));

                bias = (0 < alphas[i] && alphas[i] < regularization_param) ? b1 
                     : (0 < alphas[j] && alphas[j] < regularization_param) ? b2 
                     : (b1 + b2) / 2;

                ++alpha_changed;
            }
        }
        if (alpha_changed == 0) ++iteration;
    }
}

// Fits the SVM model to the training data.
void SVM::fit(const Eigen::MatrixXd& X, const Eigen::VectorXi& y) {
    support_vectors = X;
    optimize(X, y);
}

// Predicts the class label for a single input vector.
double SVM::predict(const Eigen::VectorXd& x) const {
    return decision_function(x) >= 0 ? 1.0 : -1.0;
}

// Predicts class labels for multiple input vectors.
Eigen::VectorXi SVM::predict(const Eigen::MatrixXd& X) const {
    Eigen::VectorXi predictions(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        predictions[i] = predict(X.row(i));
    }
    return predictions;
}

// Retrieves the support vectors used in the model.
Eigen::MatrixXd SVM::get_support_vectors() const {
    return support_vectors;
}
