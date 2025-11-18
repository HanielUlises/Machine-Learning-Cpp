#pragma once
#include <limits>

#include "kd_tree.h"

template<typename P, std::size_t D>
kd_tree<P, D>::kd_tree(const std::vector<P>* pts)
    : pts_(pts)
{
    std::vector<std::size_t> idxs(pts->size());
    for (std::size_t i = 0; i < idxs.size(); ++i) idxs[i] = i;
    nodes_.reserve(idxs.size());
    build(idxs, 0);
}

template<typename P, std::size_t D>
std::size_t kd_tree<P, D>::build(std::vector<std::size_t>& idxs, std::size_t depth) {
    if (idxs.empty()) return std::size_t(-1);

    std::size_t axis = depth % D;
    std::size_t mid = idxs.size() / 2;

    std::nth_element(
        idxs.begin(), idxs.begin() + mid, idxs.end(),
        [&](std::size_t a, std::size_t b) {
            return coord((*pts_)[a], axis) < coord((*pts_)[b], axis);
        }
    );

    std::size_t node_id = nodes_.size();
    nodes_.push_back({ idxs[mid], std::size_t(-1), std::size_t(-1), axis });

    std::vector<std::size_t> left(idxs.begin(), idxs.begin() + mid);
    std::vector<std::size_t> right(idxs.begin() + mid + 1, idxs.end());

    nodes_[node_id].left  = build(left,  depth + 1);
    nodes_[node_id].right = build(right, depth + 1);

    return node_id;
}

template<typename P, std::size_t D>
void kd_tree<P, D>::radius_recurse(std::size_t ni,
                                   const P& q,
                                   double eps_sq,
                                   std::vector<std::size_t>& out) const
{
    if (ni == std::size_t(-1)) return;

    const node& n = nodes_[ni];
    const P& p = (*pts_)[n.index];

    if (dist_sq(q, p) <= eps_sq)
        out.push_back(n.index);

    double qv = coord(q, n.axis);
    double pv = coord(p, n.axis);
    double diff = qv - pv;

    double diff_sq = diff * diff;

    if (qv < pv) {
        radius_recurse(n.left, q, eps_sq, out);
        if (diff_sq <= eps_sq)
            radius_recurse(n.right, q, eps_sq, out);
    } else {
        radius_recurse(n.right, q, eps_sq, out);
        if (diff_sq <= eps_sq)
            radius_recurse(n.left, q, eps_sq, out);
    }
}

template<typename P, std::size_t D>
std::vector<std::size_t> kd_tree<P, D>::radius_search(const P& q, double eps) const {
    double eps_sq = eps * eps;
    std::vector<std::size_t> out;
    out.reserve(32);
    if (!nodes_.empty())
        radius_recurse(0, q, eps_sq, out);
    return out;
}
