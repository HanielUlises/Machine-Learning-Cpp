#pragma once

#include <vector>
#include <cstddef>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace mlpp::model_validation {

/**
 * ROC curve and AUC for binary classifiers.
 *
 * Operates on raw classifier scores (decision values or probability estimates)
 * rather than hard predictions, so it is threshold-agnostic.  The curve is
 * built by sorting samples in descending score order and sweeping the decision
 * threshold from +∞ to -∞, recording (FPR, TPR) at every unique score value.
 *
 * Tie handling: all samples sharing the same score are processed as a batch
 * before the curve point is emitted.  This matches scikit-learn's convention
 * and avoids artificially jagged curves on low-cardinality score distributions
 * (e.g. a classifier returning only {0, 0.5, 1}).
 *
 * AUC is computed via the trapezoidal rule over the emitted (FPR, TPR) points.
 * For continuous scores with no ties this is exact; for discrete scores it is
 * a linear interpolation between breakpoints.
 *
 * Equivalence to Wilcoxon–Mann–Whitney:
 *   AUC = P(score(pos) > score(neg))
 * This interpretation holds regardless of class imbalance and is why AUC is
 * preferred over accuracy or F1 for imbalanced evaluation.
 *
 * Multiclass is supported via one-vs-rest: compute one ROC per class treating
 * that class as positive and all others as negative.  Use roc_ovr() for this.
 *
 * @tparam Score  Floating-point type of the classifier output (float, double).
 * @tparam Label  Integer-like binary label type.  Positive class = 1, negative = 0.
 */
template <typename Score = double, typename Label = int>
class ROCCurve {
public:
    struct Point {
        double fpr;  // False positive rate = FP / (FP + TN)
        double tpr;  // True  positive rate = TP / (TP + FN)
    };

    /**
     * @brief Compute the ROC curve for a single binary classification problem.
     *
     * @param scores      Raw classifier output, length n_samples.
     *                    Higher score = more likely to be the positive class.
     * @param labels      Ground-truth binary labels (positive_class or anything else).
     * @param pos_label   Value in labels that denotes the positive class (default: 1).
     */
    explicit ROCCurve(const std::vector<Score>& scores,
                      const std::vector<Label>& labels,
                      Label                     pos_label = Label(1));

    // Ordered curve points from (0,0) to (1,1).
    [[nodiscard]] const std::vector<Point>& curve()  const noexcept { return curve_; }

    // Area under the ROC curve via the trapezoidal rule.  Range [0, 1].
    [[nodiscard]] double                    auc()     const noexcept { return auc_; }

    // Number of positive samples.
    [[nodiscard]] std::size_t               n_pos()   const noexcept { return n_pos_; }

    // Number of negative samples.
    [[nodiscard]] std::size_t               n_neg()   const noexcept { return n_neg_; }

    /**
     * @brief Score at the point on the curve closest to (FPR=0, TPR=1) by
     *        Euclidean distance
     *
     * This is the Youden J statistic maximiser: argmax_t (TPR(t) - FPR(t)).
     * It minimises the distance to the top-left corner of the ROC space.
     */
    [[nodiscard]] Score optimal_threshold() const noexcept { return optimal_threshold_; }

    /**
     * @brief One-vs-rest ROC curves for a multiclass problem.
     *
     * @param scores     Score matrix, shape (n_samples, n_classes).
     *                   scores[i][k] is the model's confidence that sample i
     *                   belongs to class k.
     * @param labels     Integer class labels, length n_samples.
     * @param n_classes  Number of classes K.
     * @return           Vector of K ROCCurve objects, one per class.
     */
    [[nodiscard]] static std::vector<ROCCurve>
    roc_ovr(const std::vector<std::vector<Score>>& scores,
            const std::vector<Label>&              labels,
            std::size_t                            n_classes);

    /**
     * @brief Macro-average AUC: unweighted mean of per-class AUCs from OvR.
     */
    [[nodiscard]] static double
    macro_auc(const std::vector<std::vector<Score>>& scores,
              const std::vector<Label>&              labels,
              std::size_t                            n_classes);

private:
    std::vector<Point> curve_;
    double             auc_               = 0.0;
    std::size_t        n_pos_             = 0;
    std::size_t        n_neg_             = 0;
    Score              optimal_threshold_ = Score(0);

    void build(const std::vector<Score>& scores,
               const std::vector<Label>& labels,
               Label                     pos_label);

    // Trapezoidal rule: Σ (x[i+1] - x[i]) * (y[i+1] + y[i]) / 2
    [[nodiscard]] static double trapz(const std::vector<Point>& pts) noexcept;
};

} // namespace mlpp::model_validation

#include "roc_curve.inl"