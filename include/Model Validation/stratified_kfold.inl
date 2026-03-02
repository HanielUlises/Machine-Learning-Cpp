#pragma once

#include "stratified_kfold.hpp"

#include <unordered_map>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <random>

namespace mlpp::model_validation {

template <typename Label>
StratifiedKFold<Label>::StratifiedKFold(std::size_t n_splits,
                                        bool        shuffle,
                                        std::size_t seed)
    : n_splits_(n_splits)
    , shuffle_(shuffle)
    , seed_(seed)
{
    if (n_splits_ < 2)
        throw std::invalid_argument("StratifiedKFold: n_splits must be >= 2.");
}

template <typename Label>
std::vector<typename StratifiedKFold<Label>::Split>
StratifiedKFold<Label>::split(const std::vector<Label>& labels) const
{
    if (labels.empty())
        throw std::invalid_argument("split(): labels must be non-empty.");

    const std::size_t n = labels.size();
    auto groups = group_by_class(labels);
    n_classes_  = groups.size();

    // Validate that every class has at least n_splits_ samples.
    // If a class has fewer samples than folds, some folds would receive
    // zero examples of that class, defeating the purpose of stratification.
    for (const auto& [label, indices] : groups)
        if (indices.size() < n_splits_)
            throw std::invalid_argument(
                "split(): a class has fewer samples than n_splits. "
                "Reduce n_splits or add more data for that class.");

    // Assign each sample to a fold via round-robin within its class group.
    // S sample at position p within its
    // class group is assigned to fold (p % k), so folds receive equal-sized
    // slices from every class rather than contiguous blocks.
    std::vector<std::size_t> fold_assignment(n);

    std::mt19937 rng(seed_);

    for (auto& [label, indices] : groups) {
        if (shuffle_)
            std::shuffle(indices.begin(), indices.end(), rng);

        for (std::size_t p = 0; p < indices.size(); ++p)
            fold_assignment[indices[p]] = p % n_splits_;
    }

    // Build the k splits from the fold assignment array.
    std::vector<Split> splits;
    splits.reserve(n_splits_);

    for (std::size_t f = 0; f < n_splits_; ++f) {
        Indices train, val;
        train.reserve(n - n / n_splits_);
        val.reserve(n / n_splits_);

        for (std::size_t i = 0; i < n; ++i) {
            if (fold_assignment[i] == f)
                val.push_back(i);
            else
                train.push_back(i);
        }

        splits.emplace_back(std::move(train), std::move(val));
    }

    return splits;
}

template <typename Label>
std::vector<std::pair<Label, typename StratifiedKFold<Label>::Indices>>
StratifiedKFold<Label>::group_by_class(const std::vector<Label>& labels) const
{
    // Use an ordered structure (vector of pairs) rather than unordered_map so
    // that the iteration order — and therefore fold contents — are deterministic
    // across platforms and STL implementations.
    std::vector<std::pair<Label, Indices>> groups;
    std::unordered_map<std::size_t, std::size_t> label_to_group;

    for (std::size_t i = 0; i < labels.size(); ++i) {
        // Cast label to size_t for hashing; works for any integer-like label.
        const std::size_t key = static_cast<std::size_t>(labels[i]);
        auto it = label_to_group.find(key);
        if (it == label_to_group.end()) {
            label_to_group[key] = groups.size();
            groups.push_back({labels[i], {i}});
        } else {
            groups[it->second].second.push_back(i);
        }
    }

    return groups;
}

} // namespace mlpp::model_validation