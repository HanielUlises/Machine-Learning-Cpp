#pragma once
#include <vector>
#include <cstddef>
#include <algorithm>
#include <cmath>

template<typename PointT, std::size_t Dim>
class kd_tree {
public:
    kd_tree(const std::vector<PointT>* pts);

    std::vector<std::size_t> radius_search(const PointT& q, double eps) const;

private:
    struct node {
        std::size_t index;
        std::size_t left;
        std::size_t right;
        std::size_t axis;
    };

    const std::vector<PointT>* pts_;
    std::vector<node> nodes_;

    std::size_t build(std::vector<std::size_t>& idxs, std::size_t depth);

    void radius_recurse(std::size_t ni,
                        const PointT& q,
                        double eps_sq,
                        std::vector<std::size_t>& out) const;

    static double coord(const PointT& p, std::size_t axis) {
        return p[axis];
    }

    static double dist_sq(const PointT& a, const PointT& b) {
        double s = 0.0;
        for (std::size_t i = 0; i < Dim; ++i) {
            double d = a[i] - b[i];
            s += d * d;
        }
        return s;
    }
};

#include "kd_tree.inl"
