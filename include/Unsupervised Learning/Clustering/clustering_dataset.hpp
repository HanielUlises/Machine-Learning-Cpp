#pragma once
#include <memory>
#include <vector>
#include <string>
#include <ostream>
#include "clustering_attributes.hpp"

namespace clustering {

template<Numeric T>
class Record {
public:
    explicit Record(std::shared_ptr<class Schema<T>> schema_ptr)
        : schema_(std::move(schema_ptr)) {}

    const std::shared_ptr<class Schema<T>>& schema() const { return schema_; }

    AttrValue<T>& labelValue() { return label_; }
    const AttrValue<T>& labelValue() const { return label_; }

    AttrValue<T>& idValue() { return id_; }
    const AttrValue<T>& idValue() const { return id_; }

    std::size_t get_id() const {
        if constexpr (std::is_integral_v<T>)
            return std::get<std::size_t>(id_.get());
        else
            return static_cast<std::size_t>(std::get<T>(id_.get()));
    }

    std::size_t get_label() const {
        if constexpr (std::is_integral_v<T>)
            return std::get<std::size_t>(label_.get());
        else
            return static_cast<std::size_t>(std::get<T>(label_.get()));
    }

private:
    std::shared_ptr<class Schema<T>> schema_;
    AttrValue<T> label_;
    AttrValue<T> id_;
};

template<Numeric T>
class Schema {
public:
    virtual ~Schema() = default;

    Schema<T>* clone() const {
        return new Schema<T>(*this);
    }

    std::shared_ptr<DAttrInfo<T>>& labelInfo() { return label_info_; }
    const std::shared_ptr<DAttrInfo<T>>& labelInfo() const { return label_info_; }

    std::shared_ptr<DAttrInfo<T>>& idInfo() { return id_info_; }
    const std::shared_ptr<DAttrInfo<T>>& idInfo() const { return id_info_; }

    void set_label(std::shared_ptr<Record<T>>& r, const std::string& val) {
        std::size_t idx = label_info_->add_value(val);
        r->labelValue().set(idx);
    }

    void set_id(std::shared_ptr<Record<T>>& r, const std::string& val) {
        std::size_t idx = id_info_->add_value(val);
        r->idValue().set(idx);
    }

    bool is_labelled() const {
        return label_info_ && label_info_->num_values() > 0;
    }

    virtual bool equal(const Schema<T>& o) const {
        return *label_info_ == *o.label_info_ && *id_info_ == *o.id_info_;
    }

    virtual bool equal_no_label(const Schema<T>& o) const {
        return *id_info_ == *o.id_info_;
    }

    virtual bool operator==(const Schema<T>& o) const { return equal(o); }
    virtual bool operator!=(const Schema<T>& o) const { return !equal(o); }

    virtual bool is_member(const AttrInfo<T>& info) const {
        for (const auto& attr : *this) {
            if (*attr == info) return true;
        }
        return false;
    }

protected:
    std::shared_ptr<DAttrInfo<T>> label_info_;
    std::shared_ptr<DAttrInfo<T>> id_info_;

    std::vector<std::shared_ptr<AttrInfo<T>>> attrs_;
    auto begin() const { return attrs_.begin(); }
    auto end() const { return attrs_.end(); }
};

template<Numeric T>
class Dataset {
public:
    friend std::ostream& operator<<(std::ostream& os, const Dataset<T>& ds) {
        ds.print(os);
        return os;
    }

    explicit Dataset(std::shared_ptr<Schema<T>> schema_ptr)
        : schema_(std::move(schema_ptr)) {}

    Dataset(const Dataset<T>& other)
        : schema_(other.schema_), records_(other.records_) {}

    std::size_t num_attr() const { return schema_->attrs_.size(); }
    const std::shared_ptr<Schema<T>>& schema() const { return schema_; }

    AttrValue<T>& operator()(std::size_t i, std::size_t j) { return records_[i]->labelValue(); }
    const AttrValue<T>& operator()(std::size_t i, std::size_t j) const { return records_[i]->labelValue(); }

    bool is_numeric() const {
        for (const auto& r : records_) {
            for (const auto& attr : schema_->attrs_) {
                if (attr->type() != AttrType::Continuous) return false;
            }
        }
        return true;
    }

    bool is_categorical() const {
        for (const auto& r : records_) {
            for (const auto& attr : schema_->attrs_) {
                if (attr->type() != AttrType::Discrete) return false;
            }
        }
        return true;
    }

    void save(const std::string& filename) const {
    }

    std::vector<std::size_t> get_CM() const {
        return {};
    }

    Dataset<T>& operator=(const Dataset<T>& other) {
        if (this != &other) {
            schema_ = other.schema_;
            records_ = other.records_;
        }
        return *this;
    }

protected:
    void print(std::ostream& os) const {
        os << "Dataset with " << records_.size() << " records\n";
    }

private:
    std::shared_ptr<Schema<T>> schema_;
    std::vector<std::shared_ptr<Record<T>>> records_;
};

} // namespace clustering
