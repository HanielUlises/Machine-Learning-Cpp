#pragma once
#include <vector>
#include <memory>
#include <cstddef>

namespace mlpp::classifiers::kernel
{

using Vector = std::vector<double>;

class Kernel
{
public:
    virtual ~Kernel() = default;

    [[nodiscard]]
    virtual double operator()(const Vector& x,
                              const Vector& y) const noexcept = 0;

    [[nodiscard]]
    virtual std::unique_ptr<Kernel> clone() const = 0;
};

/**
 * Type-erased value-semantic kernel handle.
 *
 *  - Copyable
 *  - Movable
 *  - Polymorphic
 */
class KernelFunction
{
public:
    KernelFunction() = default;
    KernelFunction(const KernelFunction& other);
    KernelFunction& operator=(const KernelFunction& other);

    KernelFunction(KernelFunction&&) noexcept = default;
    KernelFunction& operator=(KernelFunction&&) noexcept = default;

    explicit KernelFunction(std::unique_ptr<Kernel> impl);

    [[nodiscard]]
    double operator()(const Vector& x,
                      const Vector& y) const noexcept;

    [[nodiscard]]
    bool valid() const noexcept;

private:
    std::unique_ptr<Kernel> impl_;
};

} // namespace mlpp::classifiers::kernel

#include "Kernel.inl"
