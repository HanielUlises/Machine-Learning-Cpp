#ifndef QDA_H
#define QDA_H

#include <Eigen/Dense>
#include <vector>
#include <stdexcept>
#include <cmath>

namespace mlpp::classifiers {
/*
    Quadratic Discriminant Analysis (QDA)
    -------------------------------------

    QDA is a generative classifier that models each class as its own multivariate
    Gaussian distribution. Unlike LDA (which forces all classes to share the same
    covariance matrix), QDA allows each class to have a *different* covariance.

    This produces a *quadratic* decision boundary, making QDA more flexible than LDA
    but also more sensitive to data sparsity and covariance singularity.

    Model assumptions:
        - Each class c has mean vector μ_c and covariance Σ_c.
        - P(x|c) ~ N( μ_c , Σ_c )
        - Class priors P(c) are estimated from frequency counts.

    Prediction:
        For each sample x, QDA computes the log-likelihood:

            log p(x|c) = −1/2 * [ (x−μ_c)^T Σ_c^{-1} (x−μ_c)
                                 + log|Σ_c|
                                 + D log(2π) ]
            log posterior = log p(x|c) + log P(c)

        The predicted label is the class with the largest posterior.

    Supports:
        - Multi-class classification
        - Any scalar type compatible with Eigen (float, double)
        - Any integer label type

*/

template<typename Scalar, typename LabelIndex = int>
class QDA {
public:
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Labels = Eigen::Matrix<LabelIndex, Eigen::Dynamic, 1>;

    QDA();

    // Fit QDA model to data X (n_samples x n_features) and labels
    void fit(const Matrix& X, const Labels& labels);

    // Predict class labels for new samples
    Labels predict(const Matrix& X) const;

    // Class posterior log-likelihoods for each sample
    Matrix predict_log_likelihood(const Matrix& X) const;

    // Accessors
    int num_classes() const { return num_classes_; }
    const std::vector<Vector>& class_means() const { return means_; }
    const std::vector<Matrix>& class_covariances() const { return covariances_; }

private:
    void compute_class_means(const Matrix& X, const Labels& labels);
    void compute_class_covariances(const Matrix& X, const Labels& labels);

private:
    int num_classes_ = 0;

    std::vector<Vector> means_;        // mean per class
    std::vector<Matrix> covariances_;  // covariance matrix per class
    std::vector<Scalar> log_det_cov_;  // log|Sigma_c|
    std::vector<Matrix> inv_cov_;      // Sigma_c^{-1}

    std::vector<Scalar> class_priors_; // P(class)
};

} // namespace mlpp::classifiers

#include "QDA.inl"

#endif // QDA_H
