#pragma once
#include <string>
#include <variant>
#include <vector>
#include <cstddef>
#include <concepts>
#include <type_traits>
#include <stdexcept>

namespace mlpp::unsupervised::clustering {

template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

/// Enumeration of attribute types
enum class AttrType {
    Unknown,
    Continuous,
    Discrete
};

template<Numeric T> class DAttrInfo;
template<Numeric T> class CAttrInfo;

/// Represents a value of an attribute (continuous or discrete)
template<Numeric T>
class AttrValue {
public:
    friend class DAttrInfo<T>;
    friend class CAttrInfo<T>;

    using value_type = std::variant<T, std::size_t>;

    AttrValue() = default;
    explicit AttrValue(value_type val) : value_(std::move(val)) {}

    const value_type& get() const { return value_; }
    void set(value_type val) { value_ = std::move(val); }

private:
    value_type value_;
};

/// Abstract base class for attributes in clustering
template<Numeric T>
class AttrInfo {
public:
    AttrInfo(const std::string& name, AttrType type)
        : name_(name), type_(type) {}
    virtual ~AttrInfo() = default;

    const std::string& name() const { return name_; }
    std::string& name() { return name_; }
    AttrType type() const { return type_; }

    virtual bool operator==(const AttrInfo<T>& info) const;
    virtual bool operator!=(const AttrInfo<T>& info) const;

    virtual AttrInfo<T>* clone() const = 0;
    virtual T distance(const AttrValue<T>&, const AttrValue<T>&) const = 0;

    virtual void set_d_val(AttrValue<T>&, std::size_t) const;
    virtual std::size_t get_d_val(const AttrValue<T>&) const;
    virtual void set_c_val(AttrValue<T>&, T) const;
    virtual T get_c_val(const AttrValue<T>&) const;

    virtual void set_unknown(AttrValue<T>&) const = 0;
    virtual bool is_unknown(const AttrValue<T>&) const = 0;

    virtual DAttrInfo<T>& cast_to_d();
    virtual const DAttrInfo<T>& cast_to_d() const;
    virtual CAttrInfo<T>& cast_to_c();
    virtual const CAttrInfo<T>& cast_to_c() const;

    virtual bool can_cast_to_d() const;
    virtual bool can_cast_to_c() const;

protected:
    bool equal_shallow(const AttrInfo<T>&) const;

private:
    std::string name_;
    AttrType type_;
};

/// Continuous (numeric) attribute
template<Numeric T>
class CAttrInfo : public AttrInfo<T> {
public:
    explicit CAttrInfo(const std::string& name);

    CAttrInfo& cast_to_c() override;
    const CAttrInfo& cast_to_c() const override;
    bool can_cast_to_c() const override;

    CAttrInfo* clone() const override;
    T distance(const AttrValue<T>&, const AttrValue<T>&) const override;

    void set_c_val(AttrValue<T>&, T) const override;
    T get_c_val(const AttrValue<T>&) const override;

    void set_unknown(AttrValue<T>&) const override;
    bool is_unknown(const AttrValue<T>&) const override;

    void set_min(T);
    void set_max(T);
    T get_min() const;
    T get_max() const;

    bool equal(const AttrInfo<T>&) const;

protected:
    T min_;
    T max_;
};

/// Discrete (categorical) attribute
template<Numeric T>
class DAttrInfo : public AttrInfo<T> {
public:
    explicit DAttrInfo(const std::string& name);

    std::size_t num_values() const;
    const std::string& int_to_str(std::size_t i) const;
    std::size_t str_to_int(const std::string&) const;
    std::size_t add_value(const std::string&, bool allow_duplicate = true);
    void remove_value(const std::string&);
    void remove_value(std::size_t i);

    DAttrInfo* clone() const override;

    T distance(const AttrValue<T>&, const AttrValue<T>&) const override;

    void set_d_val(AttrValue<T>&, std::size_t) const override;
    std::size_t get_d_val(const AttrValue<T>&) const override;

    void set_unknown(AttrValue<T>&) const override;
    bool is_unknown(const AttrValue<T>&) const override;

    DAttrInfo& cast_to_d() override;
    const DAttrInfo& cast_to_d() const override;
    bool can_cast_to_d() const override;

    bool operator==(const AttrInfo<T>& info) const override;
    bool operator!=(const AttrInfo<T>& info) const override;

protected:
    bool equal(const AttrInfo<T>&) const;

    using iterator = typename std::vector<std::string>::iterator;
    using const_iterator = typename std::vector<std::string>::const_iterator;

    std::vector<std::string> values_;
};

}
