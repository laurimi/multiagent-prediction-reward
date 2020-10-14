#ifndef NAMEDTYPE_HPP
#define NAMEDTYPE_HPP
#include <ostream>

namespace npgi {
// https://www.fluentcpp.com/2016/12/08/strong-types-for-strong-interfaces/
template <typename T, typename Parameter>
class NamedType {
 public:
  explicit NamedType(T const& value) : value_(value) {}
  explicit NamedType(T&& value) : value_(std::move(value)) {}
  T& get() { return value_; }
  T const& get() const { return value_; }

 private:
  T value_;
};

}  // namespace npgi

#endif  // NAMEDTYPE_HPP
