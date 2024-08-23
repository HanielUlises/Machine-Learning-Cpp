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
    // Constructors
    Cluster(size_t id);
    Cluster(size_t id, const std::vector<T>& items);

    // Destructor
    ~Cluster() = default;

    // Copy Constructor
    Cluster(const Cluster<T>& other);

    // Move Constructor
    Cluster(Cluster<T>&& other) noexcept;

    // Copy Assignment Operator
    Cluster<T>& operator=(const Cluster<T>& other);

    // Move Assignment Operator
    Cluster<T>& operator=(Cluster<T>&& other) noexcept;

    // Add an item to the cluster
    void add_item(const T& item);

    // Remove an item from the cluster
    void remove_item(const T& item);

    // Returns the size of the cluster
    size_t size() const;

    // Returns the data of the cluster
    const std::set<T>& data() const;

    // Merge another cluster into this one
    void merge(const Cluster<T>& other);

    // Find intersection with another cluster
    Cluster<T> intersect(const Cluster<T>& other) const;

    // Find union with another cluster
    Cluster<T> unite(const Cluster<T>& other) const;

    // Get the difference with another cluster
    Cluster<T> difference(const Cluster<T>& other) const;

    // Check if an item is in the cluster
    bool contains(const T& item) const;

    // Check if the cluster is empty
    bool is_empty() const;

    // Clear all items from the cluster
    void clear();

    // Serialize the cluster
    std::string serialize() const;

    // Deserialize a cluster from a string
    static Cluster<T> deserialize(const std::string& data);

    // Print the cluster contents
    void print() const;

    // Compare two clusters for equality
    bool operator==(const Cluster<T>& other) const;

    // Compare two clusters for inequality
    bool operator!=(const Cluster<T>& other) const;

private:
    size_t _id;
    std::set<T> _items;
};

#include "cluster.tpp"

#endif 
