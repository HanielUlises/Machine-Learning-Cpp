#pragma once

#include <memory>
#include <vector>
#include <string>
#include <ostream>
#include "clustering_attributes.hpp"

namespace mlpp::unsupervised::clustering {

template<Numeric T>
class Record;

template<Numeric T>
class Schema;

template<Numeric T>
class Dataset;

template<Numeric T>
class Record {
public:
    explicit Record(std::shared_ptr<Schema<T>> schema_ptr);

    const std::shared_ptr<Schema<T>>& schema() const;

    AttrValue<T>& labelValue();
    const AttrValue<T>& labelValue() const;

    AttrValue<T>& idValue();
    const AttrValue<T>& idValue() const;

    std::size_t get_id() const;
    std::size_t get_label() const;

private:
    std::shared_ptr<Schema<T>> schema_;
    AttrValue<T> label_;
    AttrValue<T> id_;
};

template<Numeric T>
class Schema {
public:
    virtual ~Schema() = default;

    Schema<T>* clone() const;

    std::shared_ptr<DAttrInfo<T>>& labelInfo();
    const std::shared_ptr<DAttrInfo<T>>& labelInfo() const;

    std::shared_ptr<DAttrInfo<T>>& idInfo();
    const std::shared_ptr<DAttrInfo<T>>& idInfo() const;

    void set_label(std::shared_ptr<Record<T>>& r, const std::string& val);
    void set_id(std::shared_ptr<Record<T>>& r, const std::string& val);

    bool is_labelled() const;

    virtual bool equal(const Schema<T>& o) const;
    virtual bool equal_no_label(const Schema<T>& o) const;
    virtual bool operator==(const Schema<T>& o) const;
    virtual bool operator!=(const Schema<T>& o) const;

    virtual bool is_member(const AttrInfo<T>& info) const;

protected:
    std::shared_ptr<DAttrInfo<T>> label_info_;
    std::shared_ptr<DAttrInfo<T>> id_info_;
    std::vector<std::shared_ptr<AttrInfo<T>>> attrs_;
};

template<Numeric T>
class Dataset {
public:
    friend std::ostream& operator<<(std::ostream& os, const Dataset<T>& ds);

    explicit Dataset(std::shared_ptr<Schema<T>> schema_ptr);
    Dataset(const Dataset<T>& other);

    std::size_t num_attr() const;
    const std::shared_ptr<Schema<T>>& schema() const;

    AttrValue<T>& operator()(std::size_t i, std::size_t j);
    const AttrValue<T>& operator()(std::size_t i, std::size_t j) const;

    bool is_numeric() const;
    bool is_categorical() const;

    void save(const std::string& filename) const;
    mlpp::model_validation::ConfusionMatrix<> get_CM() const;

    Dataset<T>& operator=(const Dataset<T>& other);

protected:
    void print(std::ostream& os) const;

private:
    std::shared_ptr<Schema<T>> schema_;
    std::vector<std::shared_ptr<Record<T>>> records_;
};

} // namespace mlpp::unsupervised::clustering

#include "clustering_dataset.inl"