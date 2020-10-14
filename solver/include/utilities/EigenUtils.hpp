#ifndef EIGENUTILS_HPP
#define EIGENUTILS_HPP
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <cmath>
#include "Common.hpp"

namespace npgi {
namespace detail {

template <typename Derived>
typename Eigen::MatrixBase<Derived>::Index sample_from_pmf(
    const Eigen::MatrixBase<Derived>& pmf,
    const typename Derived::Scalar& random01) {
  typename Derived::Scalar cumulative(0);

  for (typename Eigen::MatrixBase<Derived>::Index i = 0; i < pmf.size(); ++i) {
    cumulative += pmf(i);
    if (cumulative >= random01) return i;
  }
  throw std::runtime_error("failed sampling!");
}

template <typename Derived>
typename Eigen::SparseMatrixBase<Derived>::Index sample_from_pmf(
    const Eigen::SparseMatrixBase<Derived>& pmf,
    const typename Derived::Scalar& random01) {
  typename Derived::Scalar cumulative(0);
  const Derived& mat(pmf.derived());
  for (typename Eigen::SparseMatrixBase<Derived>::Index i = 0; i < pmf.size();
       ++i) {
    cumulative += mat.coeff(i);
    if (cumulative >= random01) return i;
  }
  throw std::runtime_error("failed sampling!");
}

template <typename Derived>
typename Derived::Scalar entropy(const Eigen::MatrixBase<Derived>& pmf) {
  typename Derived::Scalar entropy(0);
  for (typename Eigen::MatrixBase<Derived>::Index i = 0; i < pmf.size(); ++i) {
    if (pmf(i) > 0.0 && pmf(i) < 1.0) entropy -= pmf(i) * std::log2(pmf(i));
  }
  return entropy;
}

template <typename Derived>
typename Derived::Scalar entropy(const Eigen::SparseMatrixBase<Derived>& pmf) {
  typename Derived::Scalar entropy(0);
  for (typename Derived::InnerIterator it(pmf.derived()); it; ++it) {
    if (is_almost_zero(1.0 - it.value())) return 0.0;
    entropy -= it.value() * std::log2(it.value());
  }
  return entropy;
}

// Bayes rule application, returns prior probability.
template <typename Derived, typename OtherDerived>
typename Derived::Scalar bayes_update(
    Eigen::MatrixBase<Derived> const& pmf,
    const Eigen::MatrixBase<OtherDerived>& p_obs) {
  const typename Derived::Scalar prior_probability = pmf.dot(p_obs);
  if (!is_almost_zero(prior_probability))
    const_cast<Eigen::MatrixBase<Derived>&>(pmf) =
        pmf.cwiseProduct(p_obs) / prior_probability;
  return prior_probability;
}

template <typename Derived, typename OtherDerived>
typename Derived::Scalar bayes_update(
    Eigen::MatrixBase<Derived> const& pmf,
    const Eigen::SparseMatrixBase<OtherDerived>& p_obs) {
  const typename Derived::Scalar prior_probability = p_obs.dot(pmf);
  if (!is_almost_zero(prior_probability))
    const_cast<Eigen::MatrixBase<Derived>&>(pmf) =
        p_obs.cwiseProduct(pmf).eval() / prior_probability;
  return prior_probability;
}

template <typename Derived, typename OtherDerived>
typename Derived::Scalar bayes_update(
    Eigen::SparseMatrixBase<Derived> const& pmf,
    const Eigen::SparseMatrixBase<OtherDerived>& p_obs) {
  const typename Derived::Scalar prior_probability = pmf.dot(p_obs);
  if (!is_almost_zero(prior_probability))
    const_cast<Eigen::SparseMatrixBase<Derived>&>(pmf) =
        pmf.cwiseProduct(p_obs) / prior_probability;
  return prior_probability;
}

template <typename Derived, typename OtherDerived>
typename Derived::Scalar bayes_update(
    Eigen::SparseMatrixBase<Derived> const& pmf,
    const Eigen::MatrixBase<OtherDerived>& p_obs) {
  const typename Derived::Scalar prior_probability = pmf.dot(p_obs);
  if (!is_almost_zero(prior_probability))
    const_cast<Eigen::SparseMatrixBase<Derived>&>(pmf) =
        p_obs.cwiseProduct(pmf) / prior_probability;
  return prior_probability;
}

// in-place matrix multiplication
template <typename Derived, typename OtherDerived>
void inplace_matmul(Eigen::MatrixBase<Derived> const& x,
                    const Eigen::MatrixBase<OtherDerived>& A) {
  const_cast<Eigen::MatrixBase<Derived>&>(x) = A * x;
}

template <typename Derived, typename OtherDerived>
void inplace_matmul(Eigen::MatrixBase<Derived> const& x,
                    const Eigen::SparseMatrixBase<OtherDerived>& A) {
  const_cast<Eigen::MatrixBase<Derived>&>(x) = A * x;
}

template <typename Derived, typename OtherDerived>
void inplace_matmul(Eigen::SparseMatrixBase<Derived> const& x,
                    const Eigen::SparseMatrixBase<OtherDerived>& A) {
  const_cast<Eigen::SparseMatrixBase<Derived>&>(x) = A * x;
}

template <typename Derived, typename OtherDerived>
void inplace_matmul(Eigen::SparseMatrixBase<Derived> const& x,
                    const Eigen::MatrixBase<OtherDerived>& A) {
  typename OtherDerived::PlainObject y = (A * x);
  const_cast<Eigen::SparseMatrixBase<Derived>&>(x) = y.sparseView();
}

}  // namespace detail
}  // namespace npgi

#endif  // EIGENUTILS_HPP