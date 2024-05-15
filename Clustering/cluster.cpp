#include "Cluster.h"

// Constructor
Cluster::Cluster(size_t id) : _id(id) {}

// Adds an item to the cluster
void Cluster::add_item(size_t item) {
    _items.insert(item);
}

// Removes an item from the cluster
void Cluster::remove_item(size_t item) {
    _items.erase(item);
}

// Returns the size of the cluster
size_t Cluster::size() const {
    return _items.size();
}

// Returns the data of the cluster
const std::set<size_t>& Cluster::data() const {
    return _items;
}
