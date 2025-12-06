#ifndef LDA_H
#define LDA_H

#include <Eigen/Dense>
#include <vector>
#include <stdexcept>

// Linear Discriminant Analysis (LDA) class
// Implements LDA for dimensionality reduction and classification.
// The class computes scatter matrices, mean vectors, and projection matrix
// to transform data into a lower-dimensional space maximizing class separability.

template<typename Scalar, typename LabelIndex = int>
class LDA {
    public:
        using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
        using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
        using Labels = Eigen::Matrix<LabelIndex, Eigen::Dynamic, 1>;


        LDA();


        // Fit model to data X (n_samples x n_features) and integer labels (n_samples)
        void fit(const Matrix& X, const Labels& labels, int num_components = -1);


        // Transform data using learned projection. Returns n_samples x n_components
        Matrix transform(const Matrix& X) const;


        // Compute projection matrix explicitly. If num_components <= 0 default to num_classes - 1
        void compute_projection_matrix(int num_components = -1);


        // Accessors
        const Matrix& projection_matrix() const { return projection_matrix_; }
        const Matrix& mean_vectors() const { return mean_vectors_; }
        int num_classes() const { return num_classes_; }


        private:
        void compute_mean_vectors(const Matrix& X, const Labels& labels);
        void compute_scatter_matrices(const Matrix& X, const Labels& labels);


    private:
        int num_classes_ = 0;
        Matrix mean_vectors_; // n_features x num_classes_
        Matrix within_class_scatter_matrix_; // n_features x n_features
        Matrix between_class_scatter_matrix_; // n_features x n_features
        Matrix projection_matrix_; // n_features x n_components
        Matrix training_data_sample_;
};

#endif