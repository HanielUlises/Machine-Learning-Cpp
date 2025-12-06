// decision_tree.inl

#include "decision_tree.h"

namespace decision_trees {

namespace {

inline bool all_labels_same(const std::vector<double>& y,
                            const std::vector<std::size_t>& indices) {
    if (indices.empty()) return true;
    double first = y[indices[0]];
    return std::all_of(indices.begin(), indices.end(),
                       [&y, first](std::size_t i) { return y[i] == first; });
}

inline double gini_impurity(const std::vector<double>& y,
                            const std::vector<std::size_t>& indices) {
    if (indices.size() <= 1) return 0.0;
    std::unordered_map<double, std::size_t> counts;
    for (std::size_t i : indices) ++counts[y[i]];
    double imp = 1.0;
    for (const auto& p : counts) {
        double p_i = static_cast<double>(p.second) / indices.size();
        imp -= p_i * p_i;
    }
    return imp;
}

inline double entropy(const std::vector<double>& y,
                      const std::vector<std::size_t>& indices) {
    if (indices.size() <= 1) return 0.0;
    std::unordered_map<double, std::size_t> counts;
    for (std::size_t i : indices) ++counts[y[i]];
    double ent = 0.0;
    for (const auto& p : counts) {
        double p_i = static_cast<double>(p.second) / indices.size();
        if (p_i > 0.0) ent -= p_i * std::log2(p_i);
    }
    return ent;
}

inline double variance(const std::vector<double>& y,
                       const std::vector<std::size_t>& indices) {
    if (indices.size() <= 1) return 0.0;
    double mean = 0.0;
    for (std::size_t i : indices) mean += y[i];
    mean /= indices.size();
    double var = 0.0;
    for (std::size_t i : indices) {
        double d = y[i] - mean;
        var += d * d;
    }
    return var / indices.size();
}

inline double mean_absolute_deviation(const std::vector<double>& y,
                                      const std::vector<std::size_t>& indices) {
    if (indices.size() <= 1) return 0.0;
    std::vector<double> vals;
    vals.reserve(indices.size());
    for (std::size_t i : indices) vals.push_back(y[i]);
    std::sort(vals.begin(), vals.end());
    double median = vals[vals.size() / 2];
    double mad = 0.0;
    for (double v : vals) mad += std::abs(v - median);
    return mad / vals.size();
}

}  // anonymous namespace

DecisionTree::DecisionTree(
    Task task,
    Criterion criterion,
    std::size_t max_depth,
    std::size_t min_samples_split,
    std::size_t min_samples_leaf,
    double min_impurity_decrease)
    : task_(task),
      criterion_(criterion),
      max_depth_(max_depth),
      min_samples_split_(min_samples_split),
      min_samples_leaf_(min_samples_leaf),
      min_impurity_decrease_(min_impurity_decrease) {}

DecisionTreeClassifier::DecisionTreeClassifier(
    Criterion criterion,
    std::size_t max_depth,
    std::size_t min_samples_split,
    std::size_t min_samples_leaf,
    double min_impurity_decrease) {
    task_ = Task::classification;
    criterion_ = criterion;
    max_depth_ = max_depth;
    min_samples_split_ = min_samples_split;
    min_samples_leaf_ = min_samples_leaf;
    min_impurity_decrease_ = min_impurity_decrease;
    if (criterion_ != Criterion::gini && criterion_ != Criterion::entropy) {
        throw std::invalid_argument("Invalid criterion for classifier");
    }
}

void DecisionTreeClassifier::fit(const std::vector<std::vector<double>>& X,
                                 const std::vector<std::string>& y) {
    if (X.empty() || X.size() != y.size())
        throw std::invalid_argument("X and y size mismatch");

    class_names_.assign(y.begin(), y.end());
    std::sort(class_names_.begin(), class_names_.end());
    class_names_.erase(std::unique(class_names_.begin(), class_names_.end()), class_names_.end());

    label_to_code_.clear();
    code_to_label_.clear();
    for (std::size_t i = 0; i < class_names_.size(); ++i) {
        label_to_code_[class_names_[i]] = static_cast<double>(i);
        code_to_label_.push_back(class_names_[i]);
    }

    std::vector<double> y_num(y.size());
    for (std::size_t i = 0; i < y.size(); ++i) y_num[i] = label_to_code_.at(y[i]);

    fit(X, y_num);
}

void DecisionTreeClassifier::fit(const std::vector<std::vector<double>>& X,
                                 const std::vector<double>& y) {
    if (X.empty() || X.size() != y.size() || (not X.empty() and X[0].empty()))
        throw std::invalid_argument("Invalid input data");

    root_ = std::make_unique<TreeNode>();

    std::vector<std::size_t> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);

    build_tree(X, y, indices, 0, *root_);
}

void DecisionTreeClassifier::build_tree(const std::vector<std::vector<double>>& X,
                                        const std::vector<double>& y,
                                        const std::vector<std::size_t>& indices,
                                        std::size_t depth,
                                        TreeNode& node) {
    std::size_t n = indices.size();

    if (depth >= max_depth_ || n < min_samples_split_ || n < 2 * min_samples_leaf_ || all_labels_same(y, indices)) {
        make_leaf(node, y, indices);
        return;
    }

    double best_gain = -std::numeric_limits<double>::infinity();
    std::size_t best_feature = 0;
    double best_threshold = 0.0;
    std::vector<std::size_t> best_left, best_right;

    double parent_imp = (criterion_ == Criterion::gini) ? gini_impurity(y, indices) : entropy(y, indices);

    for (std::size_t f = 0; f < X[0].size(); ++f) {
        std::vector<std::pair<double, std::size_t>> sorted(indices.size());
        for (std::size_t j = 0; j < n; ++j) {
            std::size_t i = indices[j];
            sorted[j] = {X[i][f], i};
        }
        std::sort(sorted.begin(), sorted.end());

        for (std::size_t k = min_samples_leaf_ - 1; k + min_samples_leaf_ < n; ++k) {
            if (sorted[k].first == sorted[k + 1].first) continue;
            double thresh = (sorted[k].first + sorted[k + 1].first) / 2.0;

            std::vector<std::size_t> left_idx, right_idx;
            left_idx.reserve(k + 1);
            right_idx.reserve(n - k - 1);

            for (std::size_t j = 0; j <= k; ++j)
                left_idx.push_back(sorted[j].second);
            for (std::size_t j = k + 1; j < n; ++j)
                right_idx.push_back(sorted[j].second);

            double imp_left = (criterion_ == Criterion::gini) ? gini_impurity(y, left_idx) : entropy(y, left_idx);
            double imp_right = (criterion_ == Criterion::gini) ? gini_impurity(y, right_idx) : entropy(y, right_idx);

            double weighted = (left_idx.size() * imp_left + right_idx.size() * imp_right) / n;
            double gain = parent_imp - weighted;

            if (gain > best_gain) {
                best_gain = gain;
                best_feature = f;
                best_threshold = thresh;
                best_left = std::move(left_idx);
                best_right = std::move(right_idx);
            }
        }
    }

    if (best_gain < min_impurity_decrease_) {
        make_leaf(node, y, indices);
        return;
    }

    node.is_leaf = false;
    node.feature_index = best_feature;
    node.threshold = best_threshold;
    node.left = std::make_unique<TreeNode>();
    node.right = std::make_unique<TreeNode>();

    build_tree(X, y, best_left, depth + 1, *node.left);
    build_tree(X, y, best_right, depth + 1, *node.right);
}

void DecisionTreeClassifier::make_leaf(TreeNode& node,
                                       const std::vector<double>& y,
                                       const std::vector<std::size_t>& indices) {
    node.is_leaf = true;
    node.class_counts.assign(code_to_label_.size(), 0);

    std::unordered_map<double, std::size_t> count;
    for (std::size_t i : indices) {
        double label = y[i];
        ++count[label];
        ++node.class_counts[static_cast<std::size_t>(label)];
    }

    if (count.empty()) {
        node.value = 0.0;
        return;
    }

    auto it = std::max_element(count.begin(), count.end(),
                               [](const auto& a, const auto& b) { return a.second < b.second; });
    node.value = it->first;
    if (!code_to_label_.empty()) node.class_label = code_to_label_[static_cast<std::size_t>(node.value)];
}

double DecisionTreeClassifier::predict(const std::vector<double>& x) const {
    if (!root_) throw std::runtime_error("Tree not fitted");
    const TreeNode* n = root_.get();
    while (!n->is_leaf) {
        n = (x[n->feature_index] <= n->threshold) ? n->left.get() : n->right.get();
    }
    return n->value;
}

std::vector<double> DecisionTreeClassifier::predict(const std::vector<std::vector<double>>& X) const {
    std::vector<double> preds(X.size());
    for (std::size_t i = 0; i < X.size(); ++i) preds[i] = predict(X[i]);
    return preds;
}

std::string DecisionTreeClassifier::predict_class(const std::vector<double>& x) const {
    double code = predict(x);
    return code_to_label_[static_cast<std::size_t>(code)];
}

std::vector<std::string> DecisionTreeClassifier::predict_class(const std::vector<std::vector<double>>& X) const {
    std::vector<std::string> preds(X.size());
    for (std::size_t i = 0; i < X.size(); ++i) preds[i] = predict_class(X[i]);
    return preds;
}

std::vector<double> DecisionTreeClassifier::predict_proba(const std::vector<double>& x) const {
    if (!root_) throw std::runtime_error("Tree not fitted");
    const TreeNode* n = root_.get();
    while (!n->is_leaf) {
        n = (x[n->feature_index] <= n->threshold) ? n->left.get() : n->right.get();
    }
    std::vector<double> proba(code_to_label_.size(), 0.0);
    std::size_t total = 0;
    for (std::size_t cnt : n->class_counts) total += cnt;
    if (total == 0) {
        proba[static_cast<std::size_t>(n->value)] = 1.0;
    } else {
        for (std::size_t i = 0; i < proba.size(); ++i) {
            proba[i] = static_cast<double>(n->class_counts[i]) / total;
        }
    }
    return proba;
}

DecisionTreeRegressor::DecisionTreeRegressor(
    Criterion criterion,
    std::size_t max_depth,
    std::size_t min_samples_split,
    std::size_t min_samples_leaf,
    double min_impurity_decrease) {
    task_ = Task::regression;
    criterion_ = criterion;
    max_depth_ = max_depth;
    min_samples_split_ = min_samples_split;
    min_samples_leaf_ = min_samples_leaf;
    min_impurity_decrease_ = min_impurity_decrease;
    if (criterion_ == Criterion::friedman_mse) {
        throw std::invalid_argument("friedman_mse not implemented for basic regressor");
    } else if (criterion_ != Criterion::mse && criterion_ != Criterion::mae) {
        throw std::invalid_argument("Invalid criterion for regressor");
    }
}

void DecisionTreeRegressor::fit(const std::vector<std::vector<double>>& X,
                                const std::vector<std::string>& y) {
    throw std::runtime_error("Regressor does not accept string labels");
}

void DecisionTreeRegressor::fit(const std::vector<std::vector<double>>& X,
                                const std::vector<double>& y) {
    if (X.empty() || X.size() != y.size() || (not X.empty() and X[0].empty()))
        throw std::invalid_argument("Invalid input data");

    root_ = std::make_unique<TreeNode>();

    std::vector<std::size_t> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);

    build_tree(X, y, indices, 0, *root_);
}

void DecisionTreeRegressor::build_tree(const std::vector<std::vector<double>>& X,
                                       const std::vector<double>& y,
                                       const std::vector<std::size_t>& indices,
                                       std::size_t depth,
                                       TreeNode& node) {
    std::size_t n = indices.size();

    if (depth >= max_depth_ || n < min_samples_split_ || n < 2 * min_samples_leaf_) {
        make_leaf(node, y, indices);
        return;
    }

    double best_gain = -std::numeric_limits<double>::infinity();
    std::size_t best_feature = 0;
    double best_threshold = 0.0;
    std::vector<std::size_t> best_left, best_right;

    double (*imp_func)(const std::vector<double>&, const std::vector<std::size_t>&) =
        (criterion_ == Criterion::mae) ? mean_absolute_deviation : variance;

    double parent_imp = imp_func(y, indices);

    for (std::size_t f = 0; f < X[0].size(); ++f) {
        std::vector<std::pair<double, std::size_t>> sorted(indices.size());
        for (std::size_t j = 0; j < n; ++j) {
            std::size_t i = indices[j];
            sorted[j] = {X[i][f], i};
        }
        std::sort(sorted.begin(), sorted.end());

        for (std::size_t k = min_samples_leaf_ - 1; k + min_samples_leaf_ < n; ++k) {
            if (sorted[k].first == sorted[k + 1].first) continue;
            double thresh = (sorted[k].first + sorted[k + 1].first) / 2.0;

            std::vector<std::size_t> left_idx, right_idx;
            left_idx.reserve(k + 1);
            right_idx.reserve(n - k - 1);

            for (std::size_t j = 0; j <= k; ++j)
                left_idx.push_back(sorted[j].second);
            for (std::size_t j = k + 1; j < n; ++j)
                right_idx.push_back(sorted[j].second);

            double imp_left = imp_func(y, left_idx);
            double imp_right = imp_func(y, right_idx);

            double weighted = (left_idx.size() * imp_left + right_idx.size() * imp_right) / n;
            double gain = parent_imp - weighted;

            if (gain > best_gain) {
                best_gain = gain;
                best_feature = f;
                best_threshold = thresh;
                best_left = std::move(left_idx);
                best_right = std::move(right_idx);
            }
        }
    }

    if (best_gain < min_impurity_decrease_) {
        make_leaf(node, y, indices);
        return;
    }

    node.is_leaf = false;
    node.feature_index = best_feature;
    node.threshold = best_threshold;
    node.left = std::make_unique<TreeNode>();
    node.right = std::make_unique<TreeNode>();

    build_tree(X, y, best_left, depth + 1, *node.left);
    build_tree(X, y, best_right, depth + 1, *node.right);
}

void DecisionTreeRegressor::make_leaf(TreeNode& node,
                                      const std::vector<double>& y,
                                      const std::vector<std::size_t>& indices) {
    node.is_leaf = true;
    if (indices.empty()) {
        node.value = 0.0;
        return;
    }
    double sum = 0.0;
    for (std::size_t i : indices) sum += y[i];
    node.value = sum / indices.size();
}

double DecisionTreeRegressor::predict(const std::vector<double>& x) const {
    if (!root_) throw std::runtime_error("Tree not fitted");
    const TreeNode* n = root_.get();
    while (!n->is_leaf) {
        n = (x[n->feature_index] <= n->threshold) ? n->left.get() : n->right.get();
    }
    return n->value;
}

std::vector<double> DecisionTreeRegressor::predict(const std::vector<std::vector<double>>& X) const {
    std::vector<double> preds(X.size());
    for (std::size_t i = 0; i < X.size(); ++i) preds[i] = predict(X[i]);
    return preds;
}

}