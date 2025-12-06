// decision_tree.hpp
#pragma once

#include <vector>
#include <string>
#include <memory>
#include <cstddef>
#include <limits>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace decision_trees {

struct TreeNode {
    bool is_leaf{false};
    std::size_t feature_index{0};
    double threshold{0.0};
    double value{0.0};
    std::string class_label;
    std::vector<std::size_t> class_counts;  // For classification probabilities
    std::unique_ptr<TreeNode> left;
    std::unique_ptr<TreeNode> right;

    TreeNode() = default;
    explicit TreeNode(bool leaf) : is_leaf(leaf) {}
    virtual ~TreeNode() = default;
};

class DecisionTree {
public:
    enum class Task { classification, regression };
    enum class Criterion { gini, entropy, mse, friedman_mse, mae };

    DecisionTree(
        Task task = Task::classification,
        Criterion criterion = Criterion::gini,
        std::size_t max_depth = std::numeric_limits<std::size_t>::max(),
        std::size_t min_samples_split = 2,
        std::size_t min_samples_leaf = 1,
        double min_impurity_decrease = 0.0);

    virtual ~DecisionTree() = default;

    virtual void fit(const std::vector<std::vector<double>>& X,
                     const std::vector<std::string>& y) = 0;

    virtual void fit(const std::vector<std::vector<double>>& X,
                     const std::vector<double>& y) = 0;

    virtual double predict(const std::vector<double>& x) const = 0;

    virtual std::vector<double> predict(const std::vector<std::vector<double>>& X) const = 0;

    const TreeNode* root() const noexcept { return root_.get(); }

    const std::vector<std::string>& classes() const noexcept { return code_to_label_; }

    DecisionTree(const DecisionTree&) = delete;
    DecisionTree& operator=(const DecisionTree&) = delete;
    DecisionTree(DecisionTree&&) noexcept = default;
    DecisionTree& operator=(DecisionTree&&) noexcept = default;
    

protected:
    Task task_;
    Criterion criterion_;
    std::size_t max_depth_;
    std::size_t min_samples_split_;
    std::size_t min_samples_leaf_;
    double min_impurity_decrease_;

    std::unique_ptr<TreeNode> root_;

    std::vector<std::string> class_names_;
    std::unordered_map<std::string, double> label_to_code_;
    std::vector<std::string> code_to_label_;
};

class DecisionTreeClassifier : public DecisionTree {
private:
    void build_tree(const std::vector<std::vector<double>>& X,
                    const std::vector<double>& y,
                    const std::vector<std::size_t>& indices,
                    std::size_t depth,
                    TreeNode& node);

    void make_leaf(TreeNode& node,
                   const std::vector<double>& y,
                   const std::vector<std::size_t>& indices);

public:
    DecisionTreeClassifier(
        Criterion criterion = Criterion::gini,
        std::size_t max_depth = std::numeric_limits<std::size_t>::max(),
        std::size_t min_samples_split = 2,
        std::size_t min_samples_leaf = 1,
        double min_impurity_decrease = 0.0);

    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<std::string>& y) override;

    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<double>& y) override;

    double predict(const std::vector<double>& x) const override;
    std::vector<double> predict(const std::vector<std::vector<double>>& X) const override;

    std::string predict_class(const std::vector<double>& x) const;
    std::vector<std::string> predict_class(const std::vector<std::vector<double>>& X) const;

    std::vector<double> predict_proba(const std::vector<double>& x) const;
};

class DecisionTreeRegressor : public DecisionTree {
private:
    void build_tree(const std::vector<std::vector<double>>& X,
                    const std::vector<double>& y,
                    const std::vector<std::size_t>& indices,
                    std::size_t depth,
                    TreeNode& node);

    void make_leaf(TreeNode& node,
                   const std::vector<double>& y,
                   const std::vector<std::size_t>& indices);

public:
    DecisionTreeRegressor(
        Criterion criterion = Criterion::mse,
        std::size_t max_depth = std::numeric_limits<std::size_t>::max(),
        std::size_t min_samples_split = 2,
        std::size_t min_samples_leaf = 1,
        double min_impurity_decrease = 0.0);

    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<std::string>& y) override;

    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<double>& y) override;

    double predict(const std::vector<double>& x) const override;
    std::vector<double> predict(const std::vector<std::vector<double>>& X) const override;
};

}  // namespace decision_trees