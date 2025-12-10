#include "clustering_dataset.hpp"
#include "../Model Validation/confusion_matrix.h"
#include <algorithm>
#include <stdexcept>
#include <cmath>

namespace mlpp::unsupervised::clustering {

// Record
template<Numeric T>
inline Record<T>::Record(std::shared_ptr<Schema<T>> schema_ptr)
    : schema_(std::move(schema_ptr)) {}

template<Numeric T>
inline const std::shared_ptr<Schema<T>>& Record<T>::schema() const {
    return schema_;
}

template<Numeric T>
inline AttrValue<T>& Record<T>::labelValue() { return label_; }

template<Numeric T>
inline const AttrValue<T>& Record<T>::labelValue() const { return label_; }

template<Numeric T>
inline AttrValue<T>& Record<T>::idValue() { return id_; }

template<Numeric T>
inline const AttrValue<T>& Record<T>::idValue() const { return id_; }

template<Numeric T>
inline std::size_t Record<T>::get_id() const {
    if constexpr (std::is_integral_v<T>)
        return std::get<std::size_t>(id_.get());
    else
        return static_cast<std::size_t>(std::get<T>(id_.get()));
}

template<Numeric T>
inline std::size_t Record<T>::get_label() const {
    if constexpr (std::is_integral_v<T>)
        return std::get<std::size_t>(label_.get());
    else
        return static_cast<std::size_t>(std::get<T>(label_.get()));
}

// Schema
template<Numeric T>
inline Schema<T>* Schema<T>::clone() const {
    return new Schema<T>(*this);
}

template<Numeric T>
inline std::shared_ptr<DAttrInfo<T>>& Schema<T>::labelInfo() { return label_info_; }

template<Numeric T>
inline const std::shared_ptr<DAttrInfo<T>>& Schema<T>::labelInfo() const { return label_info_; }

template<Numeric T>
inline std::shared_ptr<DAttrInfo<T>>& Schema<T>::idInfo() { return id_info_; }

template<Numeric T>
inline const std::shared_ptr<DAttrInfo<T>>& Schema<T>::idInfo() const { return id_info_; }

template<Numeric T>
inline void Schema<T>::set_label(std::shared_ptr<Record<T>>& r, const std::string& val) {
    std::size_t idx = label_info_->add_value(val);
    r->labelValue().set(idx);
}

template<Numeric T>
inline void Schema<T>::set_id(std::shared_ptr<Record<T>>& r, const std::string& val) {
    std::size_t idx = id_info_->add_value(val);
    r->idValue().set(idx);
}

template<Numeric T>
inline bool Schema<T>::is_labelled() const {
    return label_info_ && label_info_->num_values() > 0;
}

template<Numeric T>
inline bool Schema<T>::equal(const Schema<T>& o) const {
    return *label_info_ == *o.label_info_ && *id_info_ == *o.id_info_;
}

template<Numeric T>
inline bool Schema<T>::equal_no_label(const Schema<T>& o) const {
    return *id_info_ == *o.id_info_;
}

template<Numeric T>
inline bool Schema<T>::operator==(const Schema<T>& o) const { return equal(o); }

template<Numeric T>
inline bool Schema<T>::operator!=(const Schema<T>& o) const { return !equal(o); }

template<Numeric T>
inline bool Schema<T>::is_member(const AttrInfo<T>& info) const {
    return std::any_of(attrs_.begin(), attrs_.end(), [&info](const auto& attr){ return *attr == info; });
}

// Dataset
template<Numeric T>
inline Dataset<T>::Dataset(std::shared_ptr<Schema<T>> schema_ptr)
    : schema_(std::move(schema_ptr)) {}

template<Numeric T>
inline Dataset<T>::Dataset(const Dataset<T>& other)
    : schema_(other.schema_), records_(other.records_) {}

template<Numeric T>
inline std::size_t Dataset<T>::num_attr() const { return schema_->attrs_.size(); }

template<Numeric T>
inline const std::shared_ptr<Schema<T>>& Dataset<T>::schema() const { return schema_; }

template<Numeric T>
inline AttrValue<T>& Dataset<T>::operator()(std::size_t i, std::size_t /*j*/) { return records_[i]->labelValue(); }

template<Numeric T>
inline const AttrValue<T>& Dataset<T>::operator()(std::size_t i, std::size_t /*j*/) const { return records_[i]->labelValue(); }

template<Numeric T>
inline bool Dataset<T>::is_numeric() const {
    return std::all_of(schema_->attrs_.begin(), schema_->attrs_.end(),
        [](const auto& attr){ return attr->type() == AttrType::Continuous; });
}

template<Numeric T>
inline bool Dataset<T>::is_categorical() const {
    return std::all_of(schema_->attrs_.begin(), schema_->attrs_.end(),
        [](const auto& attr){ return attr->type() == AttrType::Discrete; });
}

template<Numeric T>
inline void Dataset<T>::save(const std::string& ) const {
    // TODO: implement CSV or binary save
}

template<Numeric T>
inline mlpp::model_validation::ConfusionMatrix<> Dataset<T>::get_CM() const
{
    if (!schema_ || !schema_->is_labelled())
        return mlpp::model_validation::ConfusionMatrix<>(1);

    std::size_t n_classes = schema_->labelInfo()->num_values();
    std::size_t max_cluster = 0;

    for (const auto& r : records_) {
        std::size_t id = r->get_id();
        if (id > max_cluster) max_cluster = id;
    }

    std::size_t K = n_classes;
    if (max_cluster + 1 > K) K = max_cluster + 1;

    mlpp::model_validation::ConfusionMatrix<> cm(K);

    for (const auto& r : records_)
        cm.update(r->get_label(), r->get_id());

    return cm;
}

template<Numeric T>
inline std::ostream& operator<<(std::ostream& os, const Dataset<T>& ds) {
    ds.print(os);
    return os;
}

} // namespace mlpp::unsupervised::clustering
