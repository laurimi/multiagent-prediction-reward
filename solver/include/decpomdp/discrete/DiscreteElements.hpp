#ifndef DISCRETEELEMENTS_HPP
#define DISCRETEELEMENTS_HPP
#include "core/NamedType.hpp"
#include <string>
#include <iostream>
#include <optional>

namespace npgi {

template <typename Index, typename Parameter>
struct NamedIndexable;

template <typename Index, typename Parameter>
std::ostream& operator<<(std::ostream&, const NamedIndexable<Index, Parameter>&);

template <typename Index, typename Parameter>
std::istream& operator>>(std::istream&, NamedIndexable<Index, Parameter>&);

template <typename Index, typename Parameter>
bool operator<(const NamedIndexable<Index, Parameter>&, const NamedIndexable<Index, Parameter>&);

template <typename Index, typename Parameter>
bool operator==(const NamedIndexable<Index, Parameter>&, const NamedIndexable<Index, Parameter>&);

template <typename Index, typename Parameter>
bool operator!=(const NamedIndexable<Index, Parameter>&, const NamedIndexable<Index, Parameter>&);

template <typename Index, typename Parameter>
struct NamedIndexable {
  using index_type = Index;

  NamedIndexable() : data_(Index()) {}
  NamedIndexable(Index index) : data_(index) {}
  NamedIndexable(Index index, const std::string& name)
      : data_({index, name}) {}

  Index index() const { return data_.get().index_; }
  std::string name() const { return data_.get().name_.value_or(""); }

private:
 struct Data {
   Data(Index i) : index_(i), name_() {}
   Data(Index i, const std::string& name) : index_(i), name_(name) {}

   Index index_ {};
   std::optional<std::string> name_ {};
   bool operator==(const Data& other) const
   {
   	return ((index_ == other.index_) && (name_ == other.name_));
   }
 };
 NamedType<Data, Parameter> data_;

 friend std::ostream& operator<<<Index, Parameter>(std::ostream&,
                                        const NamedIndexable<Index, Parameter>&);
 friend std::istream& operator>>
     <Index, Parameter>(std::istream&, NamedIndexable<Index, Parameter>&);
 friend bool operator< <Index, Parameter>(const NamedIndexable&, const NamedIndexable&);
 friend bool operator== <Index, Parameter>(const NamedIndexable&, const NamedIndexable&);
 friend bool operator!= <Index, Parameter>(const NamedIndexable&, const NamedIndexable&);
};

template <typename Index, typename Parameter>
std::ostream& operator<<(std::ostream& os,
                                const NamedIndexable<Index, Parameter>& x) {
	os << x.index() << ": " << x.name();
	return os;
}

template <typename Index, typename Parameter>
std::istream& operator>>(std::istream& is,
                         NamedIndexable<Index, Parameter>& x) {
 std::string line;
 std::getline(is, line);
 const std::string delim(":");
 auto found = line.find(delim);
 x.data_.get().index_ = std::stoi(line.substr(0, found));
 x.data_.get().name_ = line.substr(found+2, std::string::npos);
 return is;
}

template <typename Index, typename Parameter>
bool operator<(const NamedIndexable<Index, Parameter>& x, const NamedIndexable<Index, Parameter>& y)
{
	return (x.data_.get().index_ < y.data_.get().index_);
}

template <typename Index, typename Parameter>
bool operator==(const NamedIndexable<Index, Parameter>& x, const NamedIndexable<Index, Parameter>& y)
{
	return (x.data_.get() == y.data_.get());
}

template <typename Index, typename Parameter>
bool operator!=(const NamedIndexable<Index, Parameter>& x, const NamedIndexable<Index, Parameter>& y)
{
	return !(x == y);
}

template <typename Index = std::size_t>
using Agent = NamedIndexable<Index, struct AgentParameter>;

template <typename Index = std::size_t>
using DiscreteLocalAction = NamedIndexable<Index, struct LocalActionParameter>;

template <typename Index = std::size_t>
using DiscreteJointAction = NamedIndexable<Index, struct JointActionParameter>;

template <typename Index = std::size_t>
using DiscreteLocalObservation =
    NamedIndexable<Index, struct LocalObservationParameter>;

template <typename Index = std::size_t>
using DiscreteJointObservation =
    NamedIndexable<Index, struct JointObservationParameter>;
}  // namespace npgi

#endif  // DISCRETEELEMENTS_HPP