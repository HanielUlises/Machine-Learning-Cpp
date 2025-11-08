#include "DecisionTree.h"
#include <unordered_map>
#include <cmath>

DecisionTree::DecisionTree() {
    root = nullptr;
}

DecisionTree::~DecisionTree() {
}

void DecisionTree::train(const std::vector<std::vector<std::string>>& data) {
    // Extract features from the data
    std::vector<std::string> features;
    for (size_t i = 0; i < data[0].size() - 1; i++) {
        features.push_back(data[0][i]);
    }

    // Build the tree
    root = build_tree(data, features);
}

std::string DecisionTree::predict(const std::vector<std::string>& instance) {
    // Start at the root and traverse the tree based on the instance's features
    auto current_node = root;
    while (current_node && current_node->decision.empty()) {
        std::string feature = current_node->feature_name;
        auto it = std::find(instance.begin(), instance.end(), feature);
        if (it != instance.end()) {
            std::string value = *it;
            auto child_it = std::find_if(current_node->children.begin(), current_node->children.end(),
                                          [&value](const std::shared_ptr<TreeNode>& child) { return child->feature_name == value; });
            if (child_it != current_node->children.end()) {
                current_node = *child_it;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    return current_node ? current_node->decision : "Unknown";
}

std::shared_ptr<TreeNode> DecisionTree::build_tree(const std::vector<std::vector<std::string>>& data, 
                                                  const std::vector<std::string>& features, 
                                                  const std::shared_ptr<TreeNode>& parent) {
    // Check if all instances have the same decision
    std::string majority_decision = get_majority_decision(data);
    if (majority_decision != "") {
        auto leaf_node = std::make_shared<TreeNode>("", std::vector<std::string>(), parent);
        leaf_node->decision = majority_decision;
        return leaf_node;
    }

    // Check if there are no features left to split
    if (features.empty()) {
        auto leaf_node = std::make_shared<TreeNode>("", std::vector<std::string>(), parent);
        leaf_node->decision = majority_decision;
        return leaf_node;
    }

    // Find the best feature to split on
    std::string best_feature = find_best_feature(data, features);
    auto root_node = std::make_shared<TreeNode>(best_feature, {}, parent);

    // Get unique values for the best feature
    std::unordered_map<std::string, std::vector<std::vector<std::string>>> subsets;
    for (const auto& instance : data) {
        std::string feature_value = instance[std::distance(features.begin(), std::find(features.begin(), features.end(), best_feature))];
        subsets[feature_value].push_back(instance);
    }

    // Create child nodes for each unique feature value
    for (const auto& [value, subset] : subsets) {
        std::vector<std::string> remaining_features = features;
        remaining_features.erase(std::remove(remaining_features.begin(), remaining_features.end(), best_feature), remaining_features.end());
        auto child_node = build_tree(subset, remaining_features, root_node);
        child_node->feature_name = value;
        root_node->children.push_back(child_node);
    }

    return root_node;
}

std::string DecisionTree::find_best_feature(const std::vector<std::vector<std::string>>& data, 
                                            const std::vector<std::string>& features) {
    double base_entropy = calculate_entropy(data);
    std::string best_feature;
    double max_info_gain = -1.0;

    for (const auto& feature : features) {
        double cond_entropy = calculate_conditional_entropy(data, feature, features);
        double info_gain = base_entropy - cond_entropy;

        if (info_gain > max_info_gain) {
            max_info_gain = info_gain;
            best_feature = feature;
        }
    }

    return best_feature;
}

double DecisionTree::calculate_entropy(const std::vector<std::vector<std::string>>& data) {
    std::unordered_map<std::string, int> class_count;
    for (const auto& instance : data) {
        // label is the last element
        std::string label = instance.back(); 
        class_count[label]++;
    }

    double entropy = 0.0;
    int total_instances = data.size();
    for (const auto& [label, count] : class_count) {
        double probability = static_cast<double>(count) / total_instances;
        entropy -= probability * log2(probability);
    }
    return entropy;
}

double DecisionTree::calculate_conditional_entropy(const std::vector<std::vector<std::string>>& data, 
                                                   const std::string& feature, 
                                                   const std::vector<std::string>& features) {
    std::unordered_map<std::string, std::vector<std::vector<std::string>>> subsets;
    int feature_index = std::distance(features.begin(), std::find(features.begin(), features.end(), feature));

    for (const auto& instance : data) {
        std::string feature_value = instance[feature_index];
        subsets[feature_value].push_back(instance);
    }

    double conditional_entropy = 0.0;
    int total_instances = data.size();
    for (const auto& [value, subset] : subsets) {
        double probability = static_cast<double>(subset.size()) / total_instances;
        double entropy = calculate_entropy(subset);
        conditional_entropy += probability * entropy;
    }
    return conditional_entropy;
}

std::string DecisionTree::get_majority_decision(const std::vector<std::vector<std::string>>& data) {
    std::unordered_map<std::string, int> decision_count;
    for (const auto& instance : data) {
        // The last element is the decision
        std::string decision = instance.back(); 
        decision_count[decision]++;
    }

    std::string majority_decision;
    int max_count = 0;
    for (const auto& [decision, count] : decision_count) {
        if (count > max_count) {
            max_count = count;
            majority_decision = decision;
        }
    }

    return majority_decision;
}

std::vector<std::vector<std::string>> DecisionTree::get_subset(const std::vector<std::vector<std::string>>& data, 
                                                              const std::string& feature, 
                                                              const std::string& value) {
    std::vector<std::vector<std::string>> subset;
    int feature_index = std::distance(data[0].begin(), std::find(data[0].begin(), data[0].end(), feature));

    for (const auto& instance : data) {
        if (instance[feature_index] == value) {
            subset.push_back(instance);
        }
    }

    return subset;
}
