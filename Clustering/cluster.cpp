#include "cluster.h"

Cluster::Cluster (size_t id) : _id(id){
}

void Cluster::add_item (size_t item) {
    _items.insert(item);
}

void Cluster::remove_item(size_t item){
    _items.erase(item);
}

size_t Cluster::size() const{
    return _items.size();
}

const std::set<size_t>& Cluster::data() const{
    return _items;
}