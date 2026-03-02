#pragma once

#include "roc_curve.hpp"

#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <limits>

namespace mlpp::model_validation {

template <typename Score, typename Label>
ROCCurve<Score, Label>::ROCCurve(const std::vector<Score>& scores,
                                 const std::vector<Label>& labels,
                                 Label                     pos_label)
{
    build(scores, labels, pos_label);
}

template <typename Score, typename Label>
void ROCCurve<Score, Label>::build(const std::vector<Score>& scores,
                                   const std::vector<Label>& labels,
                                   Label                     pos_label)
{
    if (scores.size() != labels.size())
        throw std::invalid_argument("ROCCurve: scores and labels must have the same length.");
    if (scores.empty())
        throw std::invalid_argument("ROCCurve: inputs must be non-empty.");

    const std::size_t n = scores.size();

    for (std::size_t i = 0; i < n; ++i)
        if (labels[i] == pos_label) ++n_pos_; else ++n_neg_;

    if (n_pos_ == 0 || n_neg_ == 0)
        throw std::invalid_argument(
            "ROCCurve: both classes must be present in labels.");

    // For ties, positive labels are placed
    // first (secondary sort by label descending) to match the convention that
    // a perfect classifier ranks all positives above all negatives — without
    // this tie-breaking, a classifier with many tied scores would appear worse
    // than it is because negatives would be interleaved with positives at the
    // same threshold.
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        if (scores[a] != scores[b]) return scores[a] > scores[b];
        // Ties: positive label first so we give the classifier full credit
        const bool a_pos = (labels[a] == pos_label);
        const bool b_pos = (labels[b] == pos_label);
        return a_pos > b_pos;
    });

    // Sweep threshold from +∞ to -∞, emitting one curve point per unique
    // score value.  Accumulating the entire tie group before emitting avoids
    // the staircase artefact that occurs when positives and negatives at the
    // same score are processed one-by-one.
    curve_.reserve(n + 2);
    curve_.push_back({0.0, 0.0});

    const double inv_pos = 1.0 / static_cast<double>(n_pos_);
    const double inv_neg = 1.0 / static_cast<double>(n_neg_);

    std::size_t tp = 0, fp = 0;
    double      best_j    = -1.0;
    Score       best_thr  = scores[order[0]];

    std::size_t i = 0;
    while (i < n) {
        const Score thr = scores[order[i]];
        std::size_t j   = i;
        while (j < n && scores[order[j]] == thr) {
            if (labels[order[j]] == pos_label) ++tp; else ++fp;
            ++j;
        }

        const double tpr = static_cast<double>(tp) * inv_pos;
        const double fpr = static_cast<double>(fp) * inv_neg;
        curve_.push_back({fpr, tpr});

        // Track the threshold maximising Youden's J = TPR - FPR.
        const double j_stat = tpr - fpr;
        if (j_stat > best_j) {
            best_j   = j_stat;
            best_thr = thr;
        }

        i = j;
    }

    if (curve_.back().fpr < 1.0 || curve_.back().tpr < 1.0)
        curve_.push_back({1.0, 1.0});

    optimal_threshold_ = best_thr;
    auc_               = trapz(curve_);
}

template <typename Score, typename Label>
double ROCCurve<Score, Label>::trapz(const std::vector<Point>& pts) noexcept
{
    double area = 0.0;
    for (std::size_t i = 1; i < pts.size(); ++i) {
        const double dx = pts[i].fpr - pts[i - 1].fpr;
        const double y_avg = 0.5 * (pts[i].tpr + pts[i - 1].tpr);
        area += dx * y_avg;
    }
    // AUC < 0.5 means the score is inverted relative to the positive class.
    // We return the raw value so the caller can detect this; flipping scores
    // would give 1 - AUC.
    return area;
}

template <typename Score, typename Label>
std::vector<ROCCurve<Score, Label>>
ROCCurve<Score, Label>::roc_ovr(const std::vector<std::vector<Score>>& scores,
                                const std::vector<Label>&              labels,
                                std::size_t                            n_classes)
{
    if (scores.size() != labels.size())
        throw std::invalid_argument("roc_ovr(): scores and labels must have the same length.");
    if (n_classes == 0)
        throw std::invalid_argument("roc_ovr(): n_classes must be > 0.");

    const std::size_t n = labels.size();
    std::vector<ROCCurve> curves;
    curves.reserve(n_classes);

    for (std::size_t k = 0; k < n_classes; ++k) {
        // Labels are binarzied here
        // Label = 1 means "this sample belongs to class k"; 0 means anything else.
        std::vector<Score> col(n);
        std::vector<Label> bin(n);
        for (std::size_t i = 0; i < n; ++i) {
            if (scores[i].size() <= k)
                throw std::invalid_argument(
                    "roc_ovr(): scores[i] has fewer columns than n_classes.");
            col[i] = scores[i][k];
            bin[i] = static_cast<Label>(labels[i] == static_cast<Label>(k) ? 1 : 0);
        }
        curves.emplace_back(col, bin, Label(1));
    }

    return curves;
}

template <typename Score, typename Label>
double ROCCurve<Score, Label>::macro_auc(
    const std::vector<std::vector<Score>>& scores,
    const std::vector<Label>&              labels,
    std::size_t                            n_classes)
{
    const auto curves = roc_ovr(scores, labels, n_classes);
    double sum = 0.0;
    for (const auto& c : curves) sum += c.auc();
    return sum / static_cast<double>(n_classes);
}

} // namespace mlpp::model_validation