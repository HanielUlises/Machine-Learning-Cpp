#pragma once

#include <vector>
#include <cstddef>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <random>
#include <type_traits>

namespace mlpp::model_validation {

/**
 * Stratified K-Fold cross-validation splitter.
 *
 * Partitions n sample indices into k folds such that each fold contains
 * approximately the same proportion of each class as the full dataset.
 * This matters for imbalanced problems where naive k-fold can produce
 * folds with zero representation of minority classes.
 *
 * The stratification strategy is:
 *   1. Group indices by class label.
 *   2. Within each class, shuffle (if shuffle = true), then assign indices
 *      round-robin across folds — fold 0 gets indices 0, k, 2k, …;
 *      fold 1 gets indices 1, k+1, 2k+1, … etc.
 *   3. Each split returns the union of all other folds as the training set
 *      and the held-out fold as the validation set.
 *
 * This guarantees that for a class with m samples, each fold receives
 * either ⌊m/k⌋ or ⌈m/k⌉ samples — the minimum possible imbalance.
 *
 * @tparam Label  Integer-like class label type (int, std::size_t, enum, …).
 */
template <typename Label = std::size_t>
class StratifiedKFold {
public:
    using Indices = std::vector<std::size_t>;
    using Split   = std::pair<Indices, Indices>;  ///< {train indices, val indices}

    /**
     * @param n_splits  Number of folds k ≥ 2.
     * @param shuffle   Whether to shuffle within each class before assigning
     *                  to folds.  Deterministic if false.
     * @param seed      RNG seed used when shuffle = true.
     */
    explicit StratifiedKFold(std::size_t n_splits = 5,
                             bool        shuffle   = false,
                             std::size_t seed      = 0);

    /**
     * @brief Generate all k train/val index splits.
     *
     * @param labels  Class label for each sample, length n_samples.
     * @return        Vector of k {train_indices, val_indices} pairs.
     *                Indices refer to positions in `labels`.
     */
    [[nodiscard]] std::vector<Split>
    split(const std::vector<Label>& labels) const;

    /**
     * @brief Number of unique classes found in the last call to split().
     *
     * Only valid after calling split(); returns 0 otherwise.
     */
    [[nodiscard]] std::size_t n_classes() const noexcept { return n_classes_; }

    [[nodiscard]] std::size_t n_splits()  const noexcept { return n_splits_; }

private:
    std::size_t n_splits_;
    bool        shuffle_;
    std::size_t seed_;

    mutable std::size_t n_classes_ = 0;

    /// Group sample indices by class label. Returns a map from label → indices,
    /// ordered by first occurrence so output is deterministic regardless of
    /// std::unordered_map iteration order.
    [[nodiscard]] std::vector<std::pair<Label, Indices>>
    group_by_class(const std::vector<Label>& labels) const;
};

} // namespace mlpp::model_validation

#include "stratified_kfold.inl"