#pragma once

#include "defines.hpp"

// Model Validation
#include "Model Validation/confusion_matrix.h"
#include "Model Validation/metrics.h"

// Supervised Learning
#include "Supervised Learning/Classifiers/LDA.h"
#include "Supervised Learning/Classifiers/logistic_regression.h"
#include "Supervised Learning/Classifiers/MDC.h"
#include "Supervised Learning/Classifiers/SVM.h"
#include "Supervised Learning/Decision Trees/decision_tree.h"
#include "Supervised Learning/Regression/linear_regression.h"
#include "Supervised Learning/Regression/ridge_regression.h"

// Unsupervised Learning
#include "Unsupervised Learning/Clustering/clustering_attributes.hpp"
#include "Unsupervised Learning/Clustering/clustering_dataset.hpp"
#include "Unsupervised Learning/Dimensionality Reduction/PCA.h"
#include "Unsupervised Learning/Dimensionality Reduction/SVD.h"
#include "Unsupervised Learning/KD-Tree/kd_tree.h"

#ifdef MLPP_PLOT
#include "matplotlib.h"
#endif
