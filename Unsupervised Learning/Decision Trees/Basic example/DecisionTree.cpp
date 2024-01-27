#include "DecisionTree.h"

// Constructor
DecisionTree::DecisionTree() {
    // Initialize the root as a null pointer
    root = nullptr;
}

// Destructor
DecisionTree::~DecisionTree() {
}

void DecisionTree::train(const std::vector<std::vector<std::string>>& data) {
    // Extract features from the data
    std::vector<std::string> features;
    for (size_t i = 0; i < data[0].size() - 1; i++) {
        features.push_back(data[0][i]);
    }

    // Build the tree
    root = buildTree(data, features);
}

std::string DecisionTree::predict(const std::vector<std::string>& instance) {
    // Start at the root and traverse the tree based on the instance's features
    auto currentNode = root;
    while (currentNode && !currentNode->decision.empty()) {
        std::string feature = currentNode->featureName;
        auto it = std::find(instance.begin(), instance.end(), feature);
        if (it != instance.end()) {
            std::string value = *it;
            auto childIt = std::find_if(currentNode->children.begin(), currentNode->children.end(),
                                        [&value](const std::shared_ptr<TreeNode>& child) { return child->featureName == value; });
            if (childIt != currentNode->children.end()) {
                currentNode = *childIt;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    return currentNode ? currentNode->decision : "Unknown";
}

std::shared_ptr<TreeNode> DecisionTree::buildTree(const std::vector<std::vector<std::string>>& data, 
                                                  const std::vector<std::string>& features, 
                                                  const std::shared_ptr<TreeNode>& parent) {
    // PENDING
    // 
    return nullptr;
}

std::string DecisionTree::findBestFeature(const std::vector<std::vector<std::string>>& data, 
                                          const std::vector<std::string>& features) {
    // PENDING
    return "";
}

std::string DecisionTree::getMajorityDecision(const std::vector<std::vector<std::string>>& data) {
    // PENDING
    return "";
}

std::vector<std::vector<std::string>> DecisionTree::getSubset(const std::vector<std::vector<std::string>>& data, 
                                                              const std::string& feature, 
                                                              const std::string& value) {
    // PENDING
    return std::vector<std::vector<std::string>>();
}
