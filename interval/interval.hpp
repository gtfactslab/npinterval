#pragma once
#ifndef INTERVAL_HPP
#define INTERVAL_HPP

#include <boost/numeric/interval.hpp>
#include <boost/numeric/interval/arith2.hpp>
#include <cmath>
#include <algorithm>
#include <stddef.h>

// Boost interval policy: directed rounding for arithmetic, std transcendentals.
// Requires -frounding-math for the directed-rounding arithmetic to be rigorous.
namespace npinterval_detail {
    using Policy = boost::numeric::interval_lib::policies<
        boost::numeric::interval_lib::save_state<
            boost::numeric::interval_lib::rounded_transc_std<double>>,
        boost::numeric::interval_lib::checking_base<double>
    >;
    using boost_interval = boost::numeric::interval<double, Policy>;
}

// Storage: two consecutive doubles (l, u) — same layout used in NumPy arrays.
struct interval {
    double l;  // lower bound (rounded toward -∞ by directed-rounding ops)
    double u;  // upper bound (rounded toward +∞ by directed-rounding ops)
};

// Helpers to convert between storage struct and boost interval.
static inline npinterval_detail::boost_interval _to_boost(interval i) {
    return npinterval_detail::boost_interval(i.l, i.u);
}
static inline interval _from_boost(npinterval_detail::boost_interval bi) {
    interval r; r.l = bi.lower(); r.u = bi.upper(); return r;
}

// ============================================================
// ADD
// ============================================================
static inline interval interval_add(interval i1, interval i2) {
    return _from_boost(_to_boost(i1) + _to_boost(i2));
}
static inline void interval_inplace_add(interval* i1, interval i2) {
    *i1 = interval_add(*i1, i2);
}
static inline interval interval_add_scalar(interval i, double s) {
    return _from_boost(_to_boost(i) + s);
}
static inline void interval_inplace_add_scalar(interval* i, double s) {
    *i = interval_add_scalar(*i, s);
}
static inline interval interval_scalar_add(double s, interval i) {
    return _from_boost(s + _to_boost(i));
}
static inline void interval_inplace_scalar_add(double s, interval* i) {
    *i = interval_scalar_add(s, *i);
}

// ============================================================
// SUBTRACT
// ============================================================
static inline interval interval_subtract(interval i1, interval i2) {
    return _from_boost(_to_boost(i1) - _to_boost(i2));
}
static inline void interval_inplace_subtract(interval* i1, interval i2) {
    *i1 = interval_subtract(*i1, i2);
}
static inline interval interval_subtract_scalar(interval i, double s) {
    return _from_boost(_to_boost(i) - s);
}
static inline void interval_inplace_subtract_scalar(interval* i, double s) {
    *i = interval_subtract_scalar(*i, s);
}
static inline interval interval_scalar_subtract(double s, interval i) {
    return _from_boost(s - _to_boost(i));
}
static inline void interval_inplace_scalar_subtract(double s, interval* i) {
    *i = interval_scalar_subtract(s, *i);
}

// ============================================================
// MULTIPLY
// ============================================================
static inline interval interval_multiply(interval i1, interval i2) {
    return _from_boost(_to_boost(i1) * _to_boost(i2));
}
static inline void interval_inplace_multiply(interval* i1, interval i2) {
    *i1 = interval_multiply(*i1, i2);
}
static inline interval interval_multiply_scalar(interval i, double s) {
    return _from_boost(_to_boost(i) * s);
}
static inline void interval_inplace_multiply_scalar(interval* i, double s) {
    *i = interval_multiply_scalar(*i, s);
}
static inline interval interval_scalar_multiply(double s, interval i) {
    return _from_boost(s * _to_boost(i));
}
static inline void interval_inplace_scalar_multiply(double s, interval* i) {
    *i = interval_scalar_multiply(s, *i);
}

// ============================================================
// DIVIDE
// ============================================================
static inline interval interval_inverse(interval i) {
    return _from_boost(1.0 / _to_boost(i));
}
static inline interval interval_divide(interval i1, interval i2) {
    return _from_boost(_to_boost(i1) / _to_boost(i2));
}
static inline void interval_inplace_divide(interval* i1, interval i2) {
    *i1 = interval_divide(*i1, i2);
}
static inline interval interval_divide_scalar(interval i, double s) {
    return _from_boost(_to_boost(i) / s);
}
static inline void interval_inplace_divide_scalar(interval* i, double s) {
    *i = interval_divide_scalar(*i, s);
}
static inline interval interval_scalar_divide(double s, interval i) {
    return _from_boost(s / _to_boost(i));
}
static inline void interval_inplace_scalar_divide(double s, interval* i) {
    *i = interval_scalar_divide(s, *i);
}

// ============================================================
// POWER
// ============================================================
static inline interval interval_square(interval i) {
    return _from_boost(boost::numeric::square(_to_boost(i)));
}
// [a, b] ^ s for scalar s:
//   * s < 0:           recurse on −s, then invert.
//   * s is integer n:  Boost's pow(I, n) handles parity (even → "absorb sign",
//                      odd → monotone) and rounds outward.
//   * s non-integer:   x^s is real-valued only for x > 0, so the function is
//                      defined only if i.l > 0; then [a,b]^s = [a^s, b^s]
//                      (x^s is monotone on (0,∞): increasing if s>0,
//                      decreasing if s<0 — but s<0 was already peeled off).
//                      NOTE: std::pow rounds to nearest, so the result is
//                      accurate to ±1 ULP but NOT rigorously outward-rounded;
//                      this is a known gap relative to the +/-/*// ops above.
//   * non-integer s and i.l ≤ 0:  return (−∞, +∞), since x^s isn't real on
//                                  the negative side and we can't bound it.
static inline interval interval_power_scalar(interval i, double s) {
    int p = (int)std::round(s);
    if (s < 0.0) {
        return interval_inverse(interval_power_scalar(i, -s));
    }
    if (s == (double)p) {
        // Integer power — Boost handles sign correctly
        return _from_boost(boost::numeric::pow(_to_boost(i), p));
    }
    // Non-integer power: only valid for positive intervals
    if (i.l > 0.0) {
        interval r; r.l = std::pow(i.l, s); r.u = std::pow(i.u, s); return r;
    }
    interval r; r.l = -INFINITY; r.u = INFINITY; return r;
}
static inline void interval_inplace_power_scalar(interval* i, double s) {
    *i = interval_power_scalar(*i, s);
}

// ============================================================
// UNARY
// ============================================================
static inline int interval_nonzero(interval i) {
    return !(i.l == 0.0 && i.u == 0.0);
}
// −[a, b] = [−b, −a]   (negation is exact in IEEE-754, no rounding needed)
static inline interval interval_negative(interval i) {
    return _from_boost(-_to_boost(i));
}

// ============================================================
// TRANSCENDENTAL  (uses Boost for correct outward rounding)
// ============================================================
// Per-function specifics:
//   sin / cos — monotone segments split at multiples of π/2; for wide
//     intervals (spanning a quarter-period) the bound is the full [−1, 1].
//   tan       — singularities at π/2 + kπ: intervals straddling one give
//                (−∞, +∞).
//   atan      — monotone increasing on all of ℝ; range ⊂ (−π/2, π/2).
//   tanh      — monotone increasing on all of ℝ; range ⊂ (−1, 1).
//   exp       — monotone increasing; output strictly positive.
//   sqrt      — defined for x ≥ 0; negative lower bound is clipped to 0 by
//                Boost when policy is checking_base (no exception thrown).
static inline interval interval_sin(interval i) {
    return _from_boost(boost::numeric::sin(_to_boost(i)));
}
static inline interval interval_cos(interval i) {
    return _from_boost(boost::numeric::cos(_to_boost(i)));
}
static inline interval interval_tan(interval i) {
    return _from_boost(boost::numeric::tan(_to_boost(i)));
}
static inline interval interval_arctan(interval i) {
    return _from_boost(boost::numeric::atan(_to_boost(i)));
}
static inline interval interval_tanh(interval i) {
    return _from_boost(boost::numeric::tanh(_to_boost(i)));
}
static inline interval interval_exp(interval i) {
    return _from_boost(boost::numeric::exp(_to_boost(i)));
}
static inline interval interval_sqrt(interval i) {
    return _from_boost(boost::numeric::sqrt(_to_boost(i)));
}

// ============================================================
// NORM  (width = u - l)
// ============================================================
static inline double interval_norm(interval i) {
    return boost::numeric::width(_to_boost(i));
}

// ============================================================
// SET OPERATIONS
// ============================================================
// Convex hull (Boost calls it "hull"; we expose it as union):
//   [a, b] ∪̂ [c, d] = [min(a, c),  max(b, d)]
// This is the smallest interval containing both operands. It coincides with
// the true set union iff the operands overlap; otherwise it strictly
// over-approximates by including the gap between them.
static inline interval interval_union(interval i1, interval i2) {
    return _from_boost(boost::numeric::hull(_to_boost(i1), _to_boost(i2)));
}
// Intersection:
//   [a, b] ∩ [c, d] = [max(a, c),  min(b, d)]   if intervals overlap
//                   = ∅                          otherwise (returned as NaN, NaN)
// We use NaN as the "empty interval" sentinel because the dtype storage has
// no other invalid value; downstream code can test with std::isnan(l).
static inline interval interval_intersection(interval i1, interval i2) {
    auto bi1 = _to_boost(i1);
    auto bi2 = _to_boost(i2);
    if (!boost::numeric::overlap(bi1, bi2)) {
        interval r; r.l = NAN; r.u = NAN; return r;
    }
    return _from_boost(boost::numeric::intersect(bi1, bi2));
}
// Element-wise min/max of bounds (no Boost equivalent needed).
static inline interval interval_minimum(interval i1, interval i2) {
    interval r; r.l = std::fmin(i1.l, i2.l); r.u = std::fmin(i1.u, i2.u); return r;
}
static inline interval interval_maximum(interval i1, interval i2) {
    interval r; r.l = std::fmax(i1.l, i2.l); r.u = std::fmax(i1.u, i2.u); return r;
}

// ============================================================
// COMPARISON / SUBSET
// ============================================================
static inline int interval_equal(interval i1, interval i2) {
    return (i1.l == i2.l && i1.u == i2.u);
}
static inline int interval_not_equal(interval i1, interval i2) {
    return !interval_equal(i1, i2);
}
// i1 ⊆ i2   ⟺   i2.l ≤ i1.l  AND  i1.u ≤ i2.u
static inline int interval_subseteq(interval i1, interval i2) {
    return boost::numeric::subset(_to_boost(i1), _to_boost(i2));
}
// i1 ⊇ i2   ⟺   i2 ⊆ i1
static inline int interval_supseteq(interval i1, interval i2) {
    return boost::numeric::subset(_to_boost(i2), _to_boost(i1));
}
// i1 ⊂ i2  (strict)  ⟺  i1 ⊆ i2  AND  i1 ≠ i2  (at least one endpoint is
// strictly inside i2). Boost's proper_subset implements exactly this.
static inline int interval_subset(interval i1, interval i2) {
    return boost::numeric::proper_subset(_to_boost(i1), _to_boost(i2));
}
// i1 ⊃ i2  (strict)  ⟺  i2 ⊂ i1
static inline int interval_supset(interval i1, interval i2) {
    return boost::numeric::proper_subset(_to_boost(i2), _to_boost(i1));
}

#endif // INTERVAL_HPP
