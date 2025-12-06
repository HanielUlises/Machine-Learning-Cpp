#include "MDC.h"

// Computes the Euclidean distance between two instances.
// The distance is calculated as the square root of the sum of squared differences in each feature.
double calculate_distance(const instance& instance_1, const instance& instance_2) noexcept {
    double distance_squared_sum = 0.0;

    // Accumulate the squared differences for each feature of the two instances.
    for (size_t feature_index = 0; feature_index < 2; feature_index++) {
        distance_squared_sum += std::pow(instance_1.features[feature_index] - instance_2.features[feature_index], 2);
    }
    // Euclidean distance
    return std::sqrt(distance_squared_sum);
}

// Classifies a new instance by identifying the label of the nearest training instance using the minimum distance approach.
int classify(const std::vector<instance>& training_data, const instance& new_instance) noexcept {
    if (training_data.empty()) {
        std::cerr << "Error: Training data is empty." << std::endl; 
        return -1; 
    }

    int nearest_neighbor_index = 0;
    // Distance to the first instance.
    double minimum_distance = calculate_distance(training_data[0], new_instance); 

    // Iterate through the training data to find the nearest neighbor to the new instance.
    for (size_t index = 1; index < training_data.size(); index++) {
        // Distance for the current instance.
        double current_distance = calculate_distance(training_data[index], new_instance);
        
        // Check if the current distance is smaller than the previously recorded minimum distance.
        if (current_distance < minimum_distance) {
            minimum_distance = current_distance; 
            nearest_neighbor_index = static_cast<int>(index);
        }
    }
    
    return training_data[nearest_neighbor_index].label;
}
