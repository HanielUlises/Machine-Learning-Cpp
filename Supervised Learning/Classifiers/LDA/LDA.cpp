#include "LDA.h"
#include <Eigen/Dense>
#include <iostream>
#include <cassert>

using namespace Eigen;

// Constructor for the LDA class
LDA::LDA() : nClasses(0) {}

// Computes the mean vectors for each class
void LDA::computeMeanVectors(const MatrixXd& X, const VectorXi& y) {
    // Determine the number of classes based on the labels
    nClasses = y.maxCoeff() + 1; // Assuming labels are 0 to nClasses-1
    meanVectors = MatrixXd(X.cols(), nClasses); // Initialize meanVectors matrix
    
    // Compute the mean vector for each class
    for (int c = 0; c < nClasses; ++c) {
        // Extract rows belonging to class c
        MatrixXd classData = X.transpose().rowwise() - meanVectors.col(c).transpose();
        // Original orientation
        classData = classData.transpose(); 
        classData = classData.array().rowwise() * (y.array() == c).cast<double>().transpose().array();
        meanVectors.col(c) = classData.colwise().mean();
    }
}

// Computes within-class and between-class scatter matrices
void LDA::computeScatterMatrices(const MatrixXd& X, const VectorXi& y) {
    int n = X.rows(); // Number of samples
    int d = X.cols(); // Number of features

    // Initialize scatter matrices to zero
    withinClassScatterMatrix = MatrixXd::Zero(d, d);
    betweenClassScatterMatrix = MatrixXd::Zero(d, d);

    // Compute the overall mean of the dataset
    VectorXd overallMean = X.colwise().mean();

    // Compute scatter matrices
    for (int c = 0; c < nClasses; ++c) {
        // Extract samples for class c
        MatrixXd classSamples = X.array().rowwise() - meanVectors.col(c).transpose().array();
        MatrixXd classScatter = classSamples.array().square().matrix();
        
        // Accumulate within-class scatter matrix
        withinClassScatterMatrix += (classScatter.transpose() * (y.array() == c).cast<double>()).matrix();

        // Compute and accumulate between-class scatter matrix
        VectorXd meanDifference = meanVectors.col(c) - overallMean;
        betweenClassScatterMatrix += (meanDifference * meanDifference.transpose()) * ((y.array() == c).cast<double>().sum());
    }
}

// Computes the projection matrix using Eigenvalue Decomposition
void LDA::computeProjectionMatrix() {
    // Solve the generalized eigenvalue problem for S_w^(-1) * S_b
    SelfAdjointEigenSolver<MatrixXd> solver(betweenClassScatterMatrix.inverse() * withinClassScatterMatrix);

    // Extract the eigenvectors corresponding to the largest eigenvalues
    projectionMatrix = solver.eigenvectors().transpose();
}

// Fit the LDA model to the data
void LDA::fit(const MatrixXd& X, const VectorXi& y) {
    // Validate input dimensions
    assert(X.rows() == y.size() && "Number of samples in X must match size of y");
    
    // Compute class mean vectors
    computeMeanVectors(X, y);

    // Compute scatter matrices
    computeScatterMatrices(X, y);

    // Compute the projection matrix
    computeProjectionMatrix();
}

// Project the data into the lower-dimensional space
MatrixXd LDA::transform(const MatrixXd& X) const {
    // Validate input dimensions
    assert(X.cols() == meanVectors.rows() && "Feature dimension must match training data");
    
    // Apply the learned projection matrix to transform the data
    return X * projectionMatrix.transpose();
}
