#include <iostream>
#include <set>

class Cluster{
    public:
        // Constructor
        Cluster(size_t id) : _id(id) {}

        // Adds an item to the cluster
        void add_item(size_t item) {
            _items.insert(item);
        }

        // Removes an item from the cluster
        void remove_item(size_t item) {
            _items.erase(item);
        }

        // Returns the size of the cluster
        size_t size() const {
            return _items.size();
        }

        // Returns the data of the cluster
        const std::set<size_t>& data() const {
            return _items;
        }
        
    private:
        size_t _id;
        std::set<size_t> _items;
};
