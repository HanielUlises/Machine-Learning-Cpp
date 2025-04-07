#ifndef CLUSTER_H
#define CLUSTER_H

#include <iostream>
#include <set>
#include <vector>
#include <string>
#include <stdexcept>

template <typename T>
class Cluster;

template <typename T>
class Cluster {
public:
    // Constructs a cluster with a given identifier.
    Cluster(size_t id);
    
    // Constructs a cluster with a given identifier and initializes it with items.
    Cluster(size_t id, const std::vector<T>& items);

    // Default destructor.
    ~Cluster() = default;

    // Copy constructor for creating a new cluster as a copy of another.
    Cluster(const Cluster<T>& other);

    // Move constructor for transferring ownership of resources from another cluster.
    Cluster(Cluster<T>&& other) noexcept;

    // Copy assignment operator for assigning one cluster to another.
    Cluster<T>& operator=(const Cluster<T>& other);

    // Move assignment operator for transferring ownership of resources.
    Cluster<T>& operator=(Cluster<T>&& other) noexcept;

    // Adds an item to the cluster, allowing for dynamic updates.
    void add_item(const T& item);

    // Removes an item from the cluster, facilitating item management.
    void remove_item(const T& item);

    // Returns the number of items in the cluster.
    size_t size() const;

    // Returns the set of items contained in the cluster.
    const std::set<T>& data() const;

    // Merges another cluster into this one, combining their items.
    void merge(const Cluster<T>& other);

    // Finds the intersection of this cluster with another, returning common items.
    Cluster<T> intersect(const Cluster<T>& other) const;

    // Finds the union of this cluster with another, returning all unique items.
    Cluster<T> unite(const Cluster<T>& other) const;

    // Returns the difference between this cluster and another, excluding common items.
    Cluster<T> difference(const Cluster<T>& other) const;

    // Checks if a specified item is present in the cluster.
    bool contains(const T& item) const;

    // Determines if the cluster is empty, indicating no items are present.
    bool is_empty() const;

    // Clears all items from the cluster, resetting its state.
    void clear();

    // Serializes the cluster into a string format for storage or transmission.
    std::string serialize() const;

    // Deserializes a cluster from a string, reconstructing its state.
    static Cluster<T> deserialize(const std::string& data);

    // Prints the contents of the cluster to the standard output.
    void print() const;

    // Compares two clusters for equality, based on their contents.
    bool operator==(const Cluster<T>& other) const;

    // Compares two clusters for inequality.
    bool operator!=(const Cluster<T>& other) const;

private:
    size_t _id;                // Identifier for the cluster.
    std::set<T> _items;       // Set of items contained in the cluster.
};

#include "cluster.tpp"

#endif 
