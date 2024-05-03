#include <iostream>
#include <set>

class Cluster{
    public:

        Cluster(size_t id);
        void add_item(size_t item);
        void remove_item(size_t item);
        size_t size() const;
        const std::set<size_t>& data() const;
        
    private:
    
        double _id;
        std::set<size_t> _items;
};