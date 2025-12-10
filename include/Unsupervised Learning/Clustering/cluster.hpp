#ifndef MLPP_CLUSTER_HPP
#define MLPP_CLUSTER_HPP

#include <memory>
#include <vector>
#include <cstddef>

namespace mlpp::unsupervised::clustering {

template<typename T> class Dataset;

// Cluster 
template<typename DatasetT>
class Cluster {
public:
    using dataset_type = DatasetT;
    using record_type  = typename DatasetT::record_type;

    Cluster() = default;
    virtual ~Cluster() = default;

    void set_id(std::size_t id) noexcept;
    std::size_t id() const noexcept;

protected:
    std::size_t id_ = 0;
    std::vector<std::shared_ptr<record_type>> elements_;
};

// CenterCluster 
template<typename DatasetT>
class CenterCluster : public Cluster<DatasetT> {
public:
    using dataset_type = DatasetT;
    using record_type  = typename DatasetT::record_type;

    CenterCluster() = default;
    explicit CenterCluster(const std::shared_ptr<record_type>& center);

    const std::shared_ptr<record_type>& center() const noexcept;
    std::shared_ptr<record_type>& center() noexcept;

protected:
    std::shared_ptr<record_type> center_;
};

// SubspaceCluster 
template<typename DatasetT>
class SubspaceCluster : public CenterCluster<DatasetT> {
public:
    using dataset_type = DatasetT;
    using record_type  = typename DatasetT::record_type;
    using value_type   = typename DatasetT::value_type;

    explicit SubspaceCluster(const std::shared_ptr<record_type>& center);

    std::vector<value_type>& w() noexcept;
    const std::vector<value_type>& w() const noexcept;

    value_type& w(std::size_t i) noexcept;
    const value_type& w(std::size_t i) const noexcept;

protected:
    std::vector<value_type> w_;
};

} // namespace mlpp::unsupervised::clustering

#include "cluster.inl"

#endif
