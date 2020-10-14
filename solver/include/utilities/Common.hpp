#ifndef COMMON_HPP
#define COMMON_HPP
#include <limits>

namespace npgi {

template <typename Scalar>
class RewardScaler {
 public:
  RewardScaler(Scalar minimum, Scalar maximum)
      : min(minimum), max(maximum), d(maximum - minimum) {}

  Scalar operator()(const Scalar& x) const {
    const Scalar s = (x - min) / d;
    if ((std::abs(s) < std::numeric_limits<Scalar>::epsilon())) return 0.0;
    if ((std::abs(1.0 - s) < std::numeric_limits<Scalar>::epsilon()))
      return 1.0;
    return s;
  }

 private:
  Scalar min;
  Scalar max;
  Scalar d;
};

template <typename Scalar>
struct CumulativeMovingAverage {
 public:
  CumulativeMovingAverage() : CumulativeMovingAverage(0.0, 0) {}
  CumulativeMovingAverage(Scalar init_average, std::size_t init_count)
      : avg(init_average), n(init_count) {}

  Scalar average() const { return avg; }
  std::size_t count() const { return n; }

  void update(Scalar x) {
    avg = (x + static_cast<Scalar>(n) * avg) / static_cast<Scalar>(n + 1);
    ++n;
  }

 private:
  Scalar avg;
  std::size_t n;
};

template <class T>
constexpr std::string_view type_name() {
  using namespace std;
#ifdef __clang__
  string_view p = __PRETTY_FUNCTION__;
  return string_view(p.data() + 34, p.size() - 34 - 1);
#elif defined(__GNUC__)
  string_view p = __PRETTY_FUNCTION__;
#if __cplusplus < 201402
  return string_view(p.data() + 36, p.size() - 36 - 1);
#else
  return string_view(p.data() + 49, p.find(';', 49) - 49);
#endif
#elif defined(_MSC_VER)
  string_view p = __FUNCSIG__;
  return string_view(p.data() + 84, p.size() - 84 - 7);
#endif
}

template <typename T>
bool is_almost_zero(T x) {
  return (std::abs(x) < std::numeric_limits<T>::epsilon());
}

template <typename T>
T linear_index(const std::vector<T>& multi_index,
               const std::vector<std::size_t>& dims) {
  const std::size_t multi_index_dim = multi_index.size();
  const std::size_t ndims = dims.size();
  assert(multi_index_dim <= ndims);
  if (multi_index_dim > ndims)
    throw std::runtime_error("multi_index size greater than dims");

  T linear_index = 0;
  for (std::size_t i = 0; i < ndims; ++i) {
    if (multi_index_dim <= i)
      linear_index += dims[i];
    else
      linear_index += multi_index[i] * dims[i];
  }

  return linear_index;
}

template <typename T>
std::vector<T> multi_index(T linear_index,
                           const std::vector<std::size_t>& dims) {}

}  // namespace npgi

#endif  // COMMON_HPP