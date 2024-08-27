#include "SVM.h"

// Constructor
SVM::SVM(double C, KernelType kernelType, int maxIterations, double tol, double epsilon, int degree, double gamma, double coef0) 
    : C(C), kernelType(kernelType), maxIterations(maxIterations), tol(tol), epsilon(epsilon), degree(degree), gamma(gamma), coef0(coef0), bias(0.0) {}

// Kernel Function
double SVM::kernelFunction(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const {
    double result = 0.0;
    switch (kernelType) {
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

// Decision Function
double SVM::decisionFunction(const Eigen::VectorXd& x) const {
    double result = 0.0;
    for (int i = 0; i < supportVectors.rows(); ++i) {
        result += alphas[i] * weights[i] * kernelFunction(supportVectors.row(i), x);
    }
    return result + bias;
}

// Optimize the SVM
void SVM::optimize(const Eigen::MatrixXd& X, const Eigen::VectorXi& y) {
    int m = X.rows();
    alphas = Eigen::VectorXd::Zero(m);
    weights = Eigen::VectorXd::Zero(m);
    bias = 0.0;

    int iteration = 0;
    while (iteration < maxIterations) {
        int alphaChanged = 0;
        for (int i = 0; i < m; ++i) {
            double Ei = decisionFunction(X.row(i)) - y[i];
            if ((y[i] * Ei < -tol && alphas[i] < C) || (y[i] * Ei > tol && alphas[i] > 0)) {
                // Randomly select j != i
                int j = i;
                while (j == i) {
                    j = rand() % m;
                }

                double Ej = decisionFunction(X.row(j)) - y[j];
                double alpha_i_old = alphas[i];
                double alpha_j_old = alphas[j];

                // Compute L and H
                double L, H;
                if (y[i] != y[j]) {
                    L = std::max(0.0, alphas[j] - alphas[i]);
                    H = std::min(C, C + alphas[j] - alphas[i]);
                } else {
                    L = std::max(0.0, alphas[i] + alphas[j] - C);
                    H = std::min(C, alphas[i] + alphas[j]);
                }

                if (L == H)
                    continue;

                double eta = 2.0 * kernelFunction(X.row(i), X.row(j)) - kernelFunction(X.row(i), X.row(i)) - kernelFunction(X.row(j), X.row(j));
                if (eta >= 0)
                    continue;

                // Update alpha[j]
                alphas[j] -= y[j] * (Ei - Ej) / eta;

                // Clip alpha[j]
                alphas[j] = std::clamp(alphas[j], L, H);

                if (std::abs(alphas[j] - alpha_j_old) < epsilon)
                    continue;

                // Update alpha[i]
                alphas[i] += y[i] * y[j] * (alpha_j_old - alphas[j]);

                // Update bias
                double b1 = bias - Ei - y[i] * (alphas[i] - alpha_i_old) * kernelFunction(X.row(i), X.row(i)) - y[j] * (alphas[j] - alpha_j_old) * kernelFunction(X.row(i), X.row(j));
                double b2 = bias - Ej - y[i] * (alphas[i] - alpha_i_old) * kernelFunction(X.row(i), X.row(j)) - y[j] * (alphas[j] - alpha_j_old) * kernelFunction(X.row(j), X.row(j));

                bias = (0 < alphas[i] && alphas[i] < C) ? b1 : (0 < alphas[j] && alphas[j] < C) ? b2 : (b1 + b2) / 2;

                ++alphaChanged;
            }
        }
        if (alphaChanged == 0)
            ++iteration;
    }
}

// Fit the model
void SVM::fit(const Eigen::MatrixXd& X, const Eigen::VectorXi& y) {
    supportVectors = X;
    optimize(X, y);
}

// Predict the class label
double SVM::predict(const Eigen::VectorXd& x) const {
    return decisionFunction(x) >= 0 ? 1.0 : -1.0;
}

// Predict multiple samples
Eigen::VectorXi SVM::predict(const Eigen::MatrixXd& X) const {
    Eigen::VectorXi predictions(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        predictions[i] = predict(X.row(i));
    }
    return predictions;
}

// Get support vectors
Eigen::MatrixXd SVM::getSupportVectors() const {
    return supportVectors;
}
