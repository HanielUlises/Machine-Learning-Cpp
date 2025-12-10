#pragma once
#include "clustering_attributes.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace clustering {

template<Numeric T>
inline bool AttrInfo<T>::operator==(const AttrInfo<T>& info) const {
    return equal_shallow(info);
}

template<Numeric T>
inline bool AttrInfo<T>::operator!=(const AttrInfo<T>& info) const {
    return !(*this == info);
}

template<Numeric T>
inline void AttrInfo<T>::set_d_val(AttrValue<T>&, std::size_t) const {
    throw std::runtime_error("set_d_val not implemented for base AttrInfo");
}

template<Numeric T>
inline std::size_t AttrInfo<T>::get_d_val(const AttrValue<T>&) const {
    throw std::runtime_error("get_d_val not implemented for base AttrInfo");
}

template<Numeric T>
inline void AttrInfo<T>::set_c_val(AttrValue<T>&, T) const {
    throw std::runtime_error("set_c_val not implemented for base AttrInfo");
}

template<Numeric T>
inline T AttrInfo<T>::get_c_val(const AttrValue<T>&) const {
    throw std::runtime_error("get_c_val not implemented for base AttrInfo");
}

template<Numeric T>
inline DAttrInfo<T>& AttrInfo<T>::cast_to_d() { throw std::runtime_error("Cannot cast to DAttrInfo"); }

template<Numeric T>
inline const DAttrInfo<T>& AttrInfo<T>::cast_to_d() const { throw std::runtime_error("Cannot cast to DAttrInfo"); }

template<Numeric T>
inline CAttrInfo<T>& AttrInfo<T>::cast_to_c() { throw std::runtime_error("Cannot cast to CAttrInfo"); }

template<Numeric T>
inline const CAttrInfo<T>& AttrInfo<T>::cast_to_c() const { throw std::runtime_error("Cannot cast to CAttrInfo"); }

template<Numeric T>
inline bool AttrInfo<T>::can_cast_to_d() const { return false; }

template<Numeric T>
inline bool AttrInfo<T>::can_cast_to_c() const { return false; }

template<Numeric T>
inline bool AttrInfo<T>::equal_shallow(const AttrInfo<T>& other) const {
    return (name_ == other.name()) && (type_ == other.type());
}

// CAttrInfo implementations

template<Numeric T>
inline CAttrInfo<T>::CAttrInfo(const std::string& name)
    : AttrInfo<T>(name, AttrType::Continuous), min_(0), max_(0) {}

template<Numeric T>
inline CAttrInfo<T>& CAttrInfo<T>::cast_to_c() { return *this; }

template<Numeric T>
inline const CAttrInfo<T>& CAttrInfo<T>::cast_to_c() const { return *this; }

template<Numeric T>
inline bool CAttrInfo<T>::can_cast_to_c() const { return true; }

template<Numeric T>
inline CAttrInfo<T>* CAttrInfo<T>::clone() const {
    return new CAttrInfo<T>(*this);
}

template<Numeric T>
inline T CAttrInfo<T>::distance(const AttrValue<T>& a, const AttrValue<T>& b) const {
    T val1 = std::get<T>(a.get());
    T val2 = std::get<T>(b.get());
    if constexpr (std::is_integral_v<T>)
        return val1 == val2 ? 0 : 1;
    else
        return std::abs(val1 - val2);
}

template<Numeric T>
inline void CAttrInfo<T>::set_c_val(AttrValue<T>& val, T value) const {
    val.set(value);
}

template<Numeric T>
inline T CAttrInfo<T>::get_c_val(const AttrValue<T>& val) const {
    return std::get<T>(val.get());
}

template<Numeric T>
inline void CAttrInfo<T>::set_unknown(AttrValue<T>& val) const {
    if constexpr (std::is_integral_v<T>)
        val.set(T(-1));
    else
        val.set(T{}); // 0 or NaN can be used for floating point
}

template<Numeric T>
inline bool CAttrInfo<T>::is_unknown(const AttrValue<T>& val) const {
    if constexpr (std::is_integral_v<T>)
        return std::get<T>(val.get()) == T(-1);
    else
        return std::get<T>(val.get()) == T{};
}

template<Numeric T>
inline void CAttrInfo<T>::set_min(T m) { min_ = m; }

template<Numeric T>
inline void CAttrInfo<T>::set_max(T m) { max_ = m; }

template<Numeric T>
inline T CAttrInfo<T>::get_min() const { return min_; }

template<Numeric T>
inline T CAttrInfo<T>::get_max() const { return max_; }

template<Numeric T>
inline bool CAttrInfo<T>::equal(const AttrInfo<T>& info) const {
    if (!this->equal_shallow(info)) return false;
    const auto* cinfo = dynamic_cast<const CAttrInfo<T>*>(&info);
    return cinfo && (min_ == cinfo->min_) && (max_ == cinfo->max_);
}

// DAttrInfo implementations

template<Numeric T>
inline DAttrInfo<T>::DAttrInfo(const std::string& name)
    : AttrInfo<T>(name, AttrType::Discrete) {}

template<Numeric T>
inline std::size_t DAttrInfo<T>::num_values() const { return values_.size(); }

template<Numeric T>
inline const std::string& DAttrInfo<T>::int_to_str(std::size_t i) const {
    if (i >= values_.size()) throw std::out_of_range("Index out of range");
    return values_[i];
}

template<Numeric T>
inline std::size_t DAttrInfo<T>::str_to_int(const std::string& s) const {
    auto it = std::find(values_.begin(), values_.end(), s);
    if (it != values_.end()) return std::distance(values_.begin(), it);
    throw std::invalid_argument("String value not found");
}

template<Numeric T>
inline std::size_t DAttrInfo<T>::add_value(const std::string& s, bool allow_duplicate) {
    if (!allow_duplicate) {
        auto it = std::find(values_.begin(), values_.end(), s);
        if (it != values_.end()) return std::distance(values_.begin(), it);
    }
    values_.push_back(s);
    return values_.size() - 1;
}

template<Numeric T>
inline void DAttrInfo<T>::remove_value(const std::string& s) {
    values_.erase(std::remove(values_.begin(), values_.end(), s), values_.end());
}

template<Numeric T>
inline void DAttrInfo<T>::remove_value(std::size_t i) {
    if (i >= values_.size()) throw std::out_of_range("Index out of range");
    values_.erase(values_.begin() + i);
}

template<Numeric T>
inline DAttrInfo<T>* DAttrInfo<T>::clone() const {
    return new DAttrInfo<T>(*this);
}

template<Numeric T>
inline T DAttrInfo<T>::distance(const AttrValue<T>& a, const AttrValue<T>& b) const {
    return std::get<std::size_t>(a.get()) == std::get<std::size_t>(b.get()) ? T(0) : T(1);
}

template<Numeric T>
inline void DAttrInfo<T>::set_d_val(AttrValue<T>& val, std::size_t idx) const {
    val.set(idx);
}

template<Numeric T>
inline std::size_t DAttrInfo<T>::get_d_val(const AttrValue<T>& val) const {
    return std::get<std::size_t>(val.get());
}

template<Numeric T>
inline void DAttrInfo<T>::set_unknown(AttrValue<T>& val) const {
    val.set(std::size_t(-1));
}

template<Numeric T>
inline bool DAttrInfo<T>::is_unknown(const AttrValue<T>& val) const {
    return std::get<std::size_t>(val.get()) == std::size_t(-1);
}

template<Numeric T>
inline DAttrInfo<T>& DAttrInfo<T>::cast_to_d() { return *this; }

template<Numeric T>
inline const DAttrInfo<T>& DAttrInfo<T>::cast_to_d() const { return *this; }

template<Numeric T>
inline bool DAttrInfo<T>::can_cast_to_d() const { return true; }

template<Numeric T>
inline bool DAttrInfo<T>::operator==(const AttrInfo<T>& info) const {
    if (!this->equal_shallow(info)) return false;
    const auto* dinfo = dynamic_cast<const DAttrInfo<T>*>(&info);
    return dinfo && (values_ == dinfo->values_);
}

template<Numeric T>
inline bool DAttrInfo<T>::operator!=(const AttrInfo<T>& info) const {
    return !(*this == info);
}

template<Numeric T>
inline bool DAttrInfo<T>::equal(const AttrInfo<T>& info) const {
    return *this == info;
}

} // namespace clustering
