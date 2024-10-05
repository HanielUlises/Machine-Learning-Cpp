#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include <vector>
#include <string>
#include <memory>

struct TreeNode {
    std::string feature_name;
    std::vector<std::string> possible_values;
    std::shared_ptr<TreeNode> parent;
    std::vector<std::shared_ptr<TreeNode>> children;
    std::string decision;

    TreeNode(const std::string& feature, const std::vector<std::string>& values, const std::shared_ptr<TreeNode>& parent_node = nullptr)
        : feature_name(feature), possible_values(values), parent(parent_node) {}

    ~TreeNode() = default;
};

class DecisionTree {
public:
    DecisionTree();
    ~DecisionTree();

    // Trains the decision tree using the provided dataset.
    void train(const std::vector<std::vector<std::string>>& data);

    // Predicts the class label for a given instance based on the trained decision tree.
    std::string predict(const std::vector<std::string>& instance);

private:
    // Constructs the decision tree recursively from the dataset and features.
    std::shared_ptr<TreeNode> build_tree(const std::vector<std::vector<std::string>>& data,
                                         const std::vector<std::string>& features,
                                         const std::shared_ptr<TreeNode>& parent = nullptr);

    // Identifies the best feature to split the dataset based on information gain.
    std::string find_best_feature(const std::vector<std::vector<std::string>>& data,
                                  const std::vector<std::string>& features);

    // Determines the most common decision label within a subset of the data.
    std::string get_majority_decision(const std::vector<std::vector<std::string>>& data);

    // Extracts a subset of data that corresponds to a specific feature value.
    std::vector<std::vector<std::string>> get_subset(const std::vector<std::vector<std::string>>& data,
                                                     const std::string& feature,
                                                     const std::string& value);

    // Pointer to the root node of the decision tree.
    std::shared_ptr<TreeNode> root;
};

#endif  // DECISION_TREE_H
