#ifndef CLUSTER_TPP
#define CLUSTER_TPP

#include "cluster.h"

// Constructor
template <typename T>
Cluster<T>::Cluster(size_t id) : _id(id) {}

// Constructor with items
template <typename T>
Cluster<T>::Cluster(size_t id, const std::vector<T>& items) : _id(id) {
    for (const auto& item : items) {
        _items.insert(item);
    }
}

// Copy Constructor
template <typename T>
Cluster<T>::Cluster(const Cluster<T>& other) : _id(other._id), _items(other._items) {}

// Move Constructor
template <typename T>
Cluster<T>::Cluster(Cluster<T>&& other) noexcept : _id(other._id), _items(std::move(other._items)) {
    other._id = 0;
}

// Copy Assignment Operator
template <typename T>
Cluster<T>& Cluster<T>::operator=(const Cluster<T>& other) {
    if (this != &other) {
        _id = other._id;
        _items = other._items;
    }
    return *this;
}

// Move Assignment Operator
template <typename T>
Cluster<T>& Cluster<T>::operator=(Cluster<T>&& other) noexcept {
    if (this != &other) {
        _id = other._id;
        _items = std::move(other._items);
        other._id = 0;
    }
    return *this;
}

// Add an item to the cluster
template <typename T>
void Cluster<T>::add_item(const T& item) {
    _items.insert(item);
}

// Remove an item from the cluster
template <typename T>
void Cluster<T>::remove_item(const T& item) {
    auto it = _items.find(item);
    if (it != _items.end()) {
        _items.erase(it);
    } else {
        throw std::runtime_error("Item not found in the cluster.");
    }
}

// Returns the size of the cluster
template <typename T>
size_t Cluster<T>::size() const {
    return _items.size();
}

// Returns the data of the cluster
template <typename T>
const std::set<T>& Cluster<T>::data() const {
    return _items;
}

// Merge another cluster into this one
template <typename T>
void Cluster<T>::merge(const Cluster<T>& other) {
    _items.insert(other._items.begin(), other._items.end());
}

// Find intersection with another cluster
template <typename T>
Cluster<T> Cluster<T>::intersect(const Cluster<T>& other) const {
    Cluster<T> result(_id);
    for (const auto& item : _items) {
        if (other._items.find(item) != other._items.end()) {
            result.add_item(item);
        }
    }
    return result;
}

// Find union with another cluster
template <typename T>
Cluster<T> Cluster<T>::unite(const Cluster<T>& other) const {
    Cluster<T> result(_id);
    result._items = _items;
    result._items.insert(other._items.begin(), other._items.end());
    return result;
}

// Get the difference with another cluster
template <typename T>
Cluster<T> Cluster<T>::difference(const Cluster<T>& other) const {
    Cluster<T> result(_id);
    for (const auto& item : _items) {
        if (other._items.find(item) == other._items.end()) {
            result.add_item(item);
        }
    }
    return result;
}

// Check if an item is in the cluster
template <typename T>
bool Cluster<T>::contains(const T& item) const {
    return _items.find(item) != _items.end();
}

// Check if the cluster is empty
template <typename T>
bool Cluster<T>::is_empty() const {
    return _items.empty();
}

// Clear all items from the cluster
template <typename T>
void Cluster<T>::clear() {
    _items.clear();
}

// Serialize the cluster
template <typename T>
std::string Cluster<T>::serialize() const {
    std::string serialized = "Cluster ID: " + std::to_string(_id) + "\nItems: ";
    for (const auto& item : _items) {
        serialized += std::to_string(item) + " ";
    }
    return serialized;
}

// Deserialize a cluster from a string
template <typename T>
Cluster<T> Cluster<T>::deserialize(const std::string& data) {
    size_t id = 0;
    std::vector<T> items;
    // Implement parsing logic here based on your data format
    // This is a placeholder
    return Cluster<T>(id, items);
}

// Print the cluster contents
template <typename T>
void Cluster<T>::print() const {
    std::cout << "Cluster ID: " << _id << "\nItems: ";
    for (const auto& item : _items) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
}

// Compare two clusters for equality
template <typename T>
bool Cluster<T>::operator==(const Cluster<T>& other) const {
    return _id == other._id && _items == other._items;
}

// Compare two clusters for inequality
template <typename T>
bool Cluster<T>::operator!=(const Cluster<T>& other) const {
    return !(*this == other);
}

#endif // CLUSTER_TPP
