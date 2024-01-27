// DecisionTree.h

#ifndef DECISIONTREE_H
#define DECISIONTREE_H

#include <vector>
#include <string>
#include <memory>

struct TreeNode {
    std::string featureName;
    std::vector<std::string> possibleValues;
    std::shared_ptr<TreeNode> parent;
    std::vector<std::shared_ptr<TreeNode>> children;
    std::string decision;

    TreeNode(const std::string& feature, const std::vector<std::string>& values, const std::shared_ptr<TreeNode>& parentNode = nullptr)
        : featureName(feature), possibleValues(values), parent(parentNode) {}

    ~TreeNode() = default;
};

class DecisionTree {
public:
    DecisionTree();

    ~DecisionTree();

    // Method to train the decision tree
    void train(const std::vector<std::vector<std::string>>& data);

    // Method to make predictions using the trained decision tree
    std::string predict(const std::vector<std::string>& instance);

private:
    // Private method to recursively build the decision tree
    std::shared_ptr<TreeNode> buildTree(const std::vector<std::vector<std::string>>& data,
                                        const std::vector<std::string>& features,
                                        const std::shared_ptr<TreeNode>& parent = nullptr);

    // Private method to find the best feature to split on
    std::string findBestFeature(const std::vector<std::vector<std::string>>& data,
                                const std::vector<std::string>& features);

    // Private method to get the most common decision in a set of data
    std::string getMajorityDecision(const std::vector<std::vector<std::string>>& data);

    // Private method to get the subset of data that matches a specific value for a feature
    std::vector<std::vector<std::string>> getSubset(const std::vector<std::vector<std::string>>& data,
                                                    const std::string& feature,
                                                    const std::string& value);

    // Private member variable for the root of the decision tree
    std::shared_ptr<TreeNode> root;
};

#endif  // DECISIONTREE_H
