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
    double l;
    double u;
};

static inline npinterval_detail::boost_interval _to_boost(interval i) {
    return npinterval_detail::boost_interval(i.l, i.u);
}
static inline interval _from_boost(npinterval_detail::boost_interval bi) {
    interval r; r.l = bi.lower(); r.u = bi.upper(); return r;
}

// ---- add ----
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

// ---- subtract ----
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

// ---- multiply ----
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

// ---- divide ----
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

// ---- power ----
static inline interval interval_square(interval i) {
    return _from_boost(boost::numeric::square(_to_boost(i)));
}
static inline interval interval_power_scalar(interval i, double s) {
    int p = (int)std::round(s);
    if (s < 0.0) {
        return interval_inverse(interval_power_scalar(i, -s));
    }
    if (s == (double)p) {
        return _from_boost(boost::numeric::pow(_to_boost(i), p));
    }
    // Non-integer power: defined only on strictly positive intervals.
    if (i.l > 0.0) {
        interval r; r.l = std::pow(i.l, s); r.u = std::pow(i.u, s); return r;
    }
    interval r; r.l = -INFINITY; r.u = INFINITY; return r;
}
static inline void interval_inplace_power_scalar(interval* i, double s) {
    *i = interval_power_scalar(*i, s);
}

// ---- unary ----
static inline int interval_nonzero(interval i) {
    return !(i.l == 0.0 && i.u == 0.0);
}
static inline interval interval_negative(interval i) {
    return _from_boost(-_to_boost(i));
}

// ---- transcendental ----
static inline interval interval_sin(interval i)    { return _from_boost(boost::numeric::sin(_to_boost(i))); }
static inline interval interval_cos(interval i)    { return _from_boost(boost::numeric::cos(_to_boost(i))); }
static inline interval interval_tan(interval i)    { return _from_boost(boost::numeric::tan(_to_boost(i))); }
static inline interval interval_arctan(interval i) { return _from_boost(boost::numeric::atan(_to_boost(i))); }
static inline interval interval_tanh(interval i)   { return _from_boost(boost::numeric::tanh(_to_boost(i))); }
static inline interval interval_exp(interval i)    { return _from_boost(boost::numeric::exp(_to_boost(i))); }
static inline interval interval_sqrt(interval i)   { return _from_boost(boost::numeric::sqrt(_to_boost(i))); }

// Width (u - l)
static inline double interval_norm(interval i) {
    return boost::numeric::width(_to_boost(i));
}

// ---- set ops ----
static inline interval interval_union(interval i1, interval i2) {
    return _from_boost(boost::numeric::hull(_to_boost(i1), _to_boost(i2)));
}
static inline interval interval_intersection(interval i1, interval i2) {
    auto bi1 = _to_boost(i1);
    auto bi2 = _to_boost(i2);
    if (!boost::numeric::overlap(bi1, bi2)) {
        interval r; r.l = NAN; r.u = NAN; return r;
    }
    return _from_boost(boost::numeric::intersect(bi1, bi2));
}
// Elementwise min/max on the bounds.
static inline interval interval_minimum(interval i1, interval i2) {
    interval r; r.l = std::fmin(i1.l, i2.l); r.u = std::fmin(i1.u, i2.u); return r;
}
static inline interval interval_maximum(interval i1, interval i2) {
    interval r; r.l = std::fmax(i1.l, i2.l); r.u = std::fmax(i1.u, i2.u); return r;
}

// ---- comparisons ----
static inline int interval_equal(interval i1, interval i2) {
    return (i1.l == i2.l && i1.u == i2.u);
}
static inline int interval_not_equal(interval i1, interval i2) {
    return !interval_equal(i1, i2);
}
static inline int interval_subseteq(interval i1, interval i2) {
    return boost::numeric::subset(_to_boost(i1), _to_boost(i2));
}
static inline int interval_supseteq(interval i1, interval i2) {
    return boost::numeric::subset(_to_boost(i2), _to_boost(i1));
}
static inline int interval_subset(interval i1, interval i2) {
    return boost::numeric::proper_subset(_to_boost(i1), _to_boost(i2));
}
static inline int interval_supset(interval i1, interval i2) {
    return boost::numeric::proper_subset(_to_boost(i2), _to_boost(i1));
}

#endif // INTERVAL_HPP
