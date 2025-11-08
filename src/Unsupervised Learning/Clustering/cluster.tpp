#ifndef CLUSTER_TPP
#define CLUSTER_TPP

#include "cluster.h"

// Constructs a cluster with a given identifier.
template <typename T>
Cluster<T>::Cluster(size_t id) : _id(id) {}

// Constructs a cluster with a given identifier and initializes it with items.
template <typename T>
Cluster<T>::Cluster(size_t id, const std::vector<T>& items) : _id(id) {
    for (const auto& item : items) {
        _items.insert(item);
    }
}

// Copy constructor for creating a new cluster as a copy of another.
template <typename T>
Cluster<T>::Cluster(const Cluster<T>& other) : _id(other._id), _items(other._items) {}

// Move constructor for transferring ownership of resources from another cluster.
template <typename T>
Cluster<T>::Cluster(Cluster<T>&& other) noexcept : _id(other._id), _items(std::move(other._items)) {
    other._id = 0;
}

// Copy assignment operator for assigning one cluster to another.
template <typename T>
Cluster<T>& Cluster<T>::operator=(const Cluster<T>& other) {
    if (this != &other) {
        _id = other._id;
        _items = other._items;
    }
    return *this;
}

// Move assignment operator for transferring ownership of resources.
template <typename T>
Cluster<T>& Cluster<T>::operator=(Cluster<T>&& other) noexcept {
    if (this != &other) {
        _id = other._id;
        _items = std::move(other._items);
        other._id = 0;
    }
    return *this;
}

// Adds an item to the cluster, allowing for dynamic updates.
template <typename T>
void Cluster<T>::add_item(const T& item) {
    _items.insert(item);
}

// Removes an item from the cluster, facilitating item management.
template <typename T>
void Cluster<T>::remove_item(const T& item) {
    auto it = _items.find(item);
    if (it != _items.end()) {
        _items.erase(it);
    } else {
        throw std::runtime_error("Item not found in the cluster.");
    }
}

// Returns the number of items in the cluster.
template <typename T>
size_t Cluster<T>::size() const {
    return _items.size();
}

// Returns the set of items contained in the cluster.
template <typename T>
const std::set<T>& Cluster<T>::data() const {
    return _items;
}

// Merges another cluster into this one, combining their items.
template <typename T>
void Cluster<T>::merge(const Cluster<T>& other) {
    _items.insert(other._items.begin(), other._items.end());
}

// Finds the intersection of this cluster with another, returning common items.
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

// Finds the union of this cluster with another, returning all unique items.
template <typename T>
Cluster<T> Cluster<T>::unite(const Cluster<T>& other) const {
    Cluster<T> result(_id);
    result._items = _items;
    result._items.insert(other._items.begin(), other._items.end());
    return result;
}

// Returns the difference between this cluster and another, excluding common items.
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

// Checks if a specified item is present in the cluster.
template <typename T>
bool Cluster<T>::contains(const T& item) const {
    return _items.find(item) != _items.end();
}

// Determines if the cluster is empty, indicating no items are present.
template <typename T>
bool Cluster<T>::is_empty() const {
    return _items.empty();
}

// Clears all items from the cluster, resetting its state.
template <typename T>
void Cluster<T>::clear() {
    _items.clear();
}

// Serializes the cluster into a string format for storage or transmission.
template <typename T>
std::string Cluster<T>::serialize() const {
    std::string serialized = "Cluster ID: " + std::to_string(_id) + "\nItems: ";
    for (const auto& item : _items) {
        serialized += std::to_string(item) + " ";
    }
    return serialized;
}

// Deserializes a cluster from a string, reconstructing its state.
template <typename T>
Cluster<T> Cluster<T>::deserialize(const std::string& data) {
    size_t id = 0;
    std::vector<T> items;
    // This is a placeholder
    return Cluster<T>(id, items);
}

// Prints the contents of the cluster to the standard output.
template <typename T>
void Cluster<T>::print() const {
    std::cout << "Cluster ID: " << _id << "\nItems: ";
    for (const auto& item : _items) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
}

// Compares two clusters for equality, based on their contents.
template <typename T>
bool Cluster<T>::operator==(const Cluster<T>& other) const {
    return _id == other._id && _items == other._items;
}

// Compares two clusters for inequality.
template <typename T>
bool Cluster<T>::operator!=(const Cluster<T>& other) const {
    return !(*this == other);
}

#endif // CLUSTER_TPP
