#include "cluster.hpp"

namespace mlpp::unsupervised::clustering {

//  Cluster 

template<typename DatasetT>
inline void Cluster<DatasetT>::set_id(std::size_t id) noexcept {
    id_ = id;
}

template<typename DatasetT>
inline std::size_t Cluster<DatasetT>::id() const noexcept {
    return id_;
}

//  CenterCluster 

template<typename DatasetT>
inline CenterCluster<DatasetT>::CenterCluster(
    const std::shared_ptr<record_type>& center
) : center_(center)
{}

template<typename DatasetT>
inline const std::shared_ptr<typename DatasetT::record_type>&
CenterCluster<DatasetT>::center() const noexcept {
    return center_;
}

template<typename DatasetT>
inline std::shared_ptr<typename DatasetT::record_type>&
CenterCluster<DatasetT>::center() noexcept {
    return center_;
}

//  SubspaceCluster 

template<typename DatasetT>
inline SubspaceCluster<DatasetT>::SubspaceCluster(
    const std::shared_ptr<record_type>& center
) : CenterCluster<DatasetT>(center)
{}

template<typename DatasetT>
inline std::vector<typename DatasetT::value_type>&
SubspaceCluster<DatasetT>::w() noexcept {
    return w_;
}

template<typename DatasetT>
inline const std::vector<typename DatasetT::value_type>&
SubspaceCluster<DatasetT>::w() const noexcept {
    return w_;
}

template<typename DatasetT>
inline typename DatasetT::value_type&
SubspaceCluster<DatasetT>::w(std::size_t i) noexcept {
    return w_[i];
}

template<typename DatasetT>
inline const typename DatasetT::value_type&
SubspaceCluster<DatasetT>::w(std::size_t i) const noexcept {
    return w_[i];
}

} // namespace mlpp::unsupervised::clustering
