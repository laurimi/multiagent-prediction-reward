#ifndef SAMPLINGUTILITIES_HPP
#define SAMPLINGUTILITIES_HPP
#include <random>
#include <algorithm>
#include <vector>
namespace npgi {

// TODO: reservoir sampling with reservoir size K
// (* S has items to sample, R will contain the result *)
// ReservoirSample(S[1..n], R[1..k])
//   // fill the reservoir array
//   for i := 1 to k
//       R[i] := S[i]

//   // replace elements with gradually decreasing probability
//   for i := k+1 to n
//     (* randomInteger(a, b) generates a uniform integer from the inclusive
//     range {a, ..., b} *)
//     j := randomInteger(1, i)
//     if j <= k
//         R[j] := S[i]
template <typename InputIt, typename RandomNumberGenerator>
InputIt reservoir_sample(InputIt first, InputIt last,
                         RandomNumberGenerator& g) {
  std::uniform_real_distribution<> d;
  std::size_t n_items(1);
  InputIt sample(last);
  for (; first != last; ++first, ++n_items) {
    const double swap_probability = 1.0 / static_cast<double>(n_items);
    if (d(g) < swap_probability) sample = first;
  }
  return sample;
}

// "Kraemer algorithm" as explained in
// Smith & Tromble (2004): "Sampling Uniformly from the Unit Simplex"
// https://www.cs.cmu.edu/~nasmith/papers/smith+tromble.tr04.pdf
// Not perfect, but good enough
template <typename Scalar, typename RandomNumberGenerator>
std::vector<Scalar> sample_unit_simplex(std::size_t n, RandomNumberGenerator& g) {
	std::uniform_int_distribution<int> dist(0, std::numeric_limits<int>::max());
	std::vector<int> v(n+1);
	auto gen = [&dist, &g]() { return dist(g); };
	std::generate(std::begin(v), std::end(v), gen);
	std::sort(std::begin(v), std::end(v));
	v.front() = 0;
	v.back() = std::numeric_limits<int>::max();
	// y_i = x_{i} - x_{i-1} for all i = 1, ..., n
	std::transform(v.begin()+1, v.end(), v.begin(), v.begin(), [](int xi, int xm){ return (xi-xm);});

	const Scalar M = static_cast<Scalar>(std::numeric_limits<int>::max());
	std::vector<Scalar> b(v.begin(), v.end()-1);
	std::transform(b.begin(), b.end(), b.begin(), [&M](Scalar x){ return (x/M); });

	return b;
}

}  // namespace npgi

#endif  // SAMPLINGUTILITIES_HPP