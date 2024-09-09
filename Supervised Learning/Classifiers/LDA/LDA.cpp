#include "LDA.h"
#include <Eigen/Dense>
#include <iostream>

using namespace Eigen;

// Constructor
LDA::LDA() : nClasses(0) {}

// Compute mean vectors for each class
void LDA::computeMeanVectors(const MatrixXd& X, const VectorXi& y) {
    // Assuming labels are from 0 to nClasses-1
    nClasses = y.maxCoeff() + 1; 
    meanVectors = MatrixXd(X.cols(), nClasses);
    
    for (int c = 0; c < nClasses; ++c) {
        VectorXd classData = X.colwise().mean();
        meanVectors.col(c) = classData;
    }
}

// Compute within-class and between-class scatter matrices
void LDA::computeScatterMatrices(const MatrixXd& X, const VectorXi& y) {
    int n = X.rows();
    int d = X.cols();
    
    withinClassScatterMatrix = MatrixXd::Zero(d, d);
    betweenClassScatterMatrix = MatrixXd::Zero(d, d);
    
    VectorXd overallMean = X.colwise().mean();
    
    for (int c = 0; c < nClasses; ++c) {
        MatrixXd classData = X.transpose() - meanVectors.col(c).transpose();
        classData = classData.transpose().array().square().matrix();
        withinClassScatterMatrix += (classData.transpose() * (y.array() == c).cast<double>()).matrix();
        
        VectorXd meanDifference = meanVectors.col(c) - overallMean;
        betweenClassScatterMatrix += (meanDifference * meanDifference.transpose()) * ((y.array() == c).cast<double>().sum());
    }
}

// Compute the projection matrix
void LDA::computeProjectionMatrix() {
    Eigen::SelfAdjointEigenSolver<MatrixXd> solver(betweenClassScatterMatrix.inverse() * withinClassScatterMatrix);
    projectionMatrix = solver.eigenvectors().transpose();
}

// Fit the LDA model
void LDA::fit(const MatrixXd& X, const VectorXi& y) {
    computeMeanVectors(X, y);
    computeScatterMatrices(X, y);
    computeProjectionMatrix();
}

// Transform the data using the learned projection matrix
MatrixXd LDA::transform(const MatrixXd& X) const {
    return X * projectionMatrix.transpose();
}
