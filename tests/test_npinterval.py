"""Unit tests for the npinterval (numpy 2 / Boost.Interval) dtype."""

import math
import pickle

import numpy as np
import pytest

import interval as _itv  # noqa: F401  (registers np.interval)
from interval import (
    as_iarray,
    as_lu,
    from_cent_pert,
    get_cent_pert,
    get_half_intervals,
    get_iarray,
    get_lu,
    has_nan,
    is_iarray,
    one,
    zero,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_contains(iv, *values, slack=0.0):
    """Each value must lie within [l - slack, u + slack]."""
    for v in values:
        assert iv.l - slack <= v <= iv.u + slack, (
            f"{v} not in [{iv.l}, {iv.u}]"
        )


def assert_close_bounds(iv, l, u, tol=1e-12):
    assert abs(iv.l - l) <= tol, f"lower {iv.l} != {l}"
    assert abs(iv.u - u) <= tol, f"upper {iv.u} != {u}"


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_widens_by_one_ulp(self):
        a = np.interval(1.0, 2.0)
        assert a.l < 1.0 < a.u  # 1 is inside
        assert a.l < 2.0 < a.u  # 2 is inside
        # widening is by ±1 ULP exactly
        assert a.l == np.nextafter(1.0, -np.inf)
        assert a.u == np.nextafter(2.0, np.inf)

    def test_exact_no_widening(self):
        a = np.interval(1.0, 2.0, exact=True)
        assert a.l == 1.0 and a.u == 2.0

    def test_single_arg_point_widened(self):
        a = np.interval(3.0)
        assert a.l == np.nextafter(3.0, -np.inf)
        assert a.u == np.nextafter(3.0, np.inf)

    def test_single_arg_exact(self):
        a = np.interval(3.0, exact=True)
        assert a.l == 3.0 and a.u == 3.0

    def test_copy_constructor_preserves_bounds(self):
        a = np.interval(1.5, 2.5, exact=True)
        b = np.interval(a)
        assert b.l == a.l and b.u == a.u

    def test_integer_args(self):
        a = np.interval(1, 2, exact=True)
        assert a.l == 1.0 and a.u == 2.0

    def test_zero_args(self):
        a = np.interval()
        assert a.l == 0.0 and a.u == 0.0


# ---------------------------------------------------------------------------
# Arithmetic on scalar intervals
# ---------------------------------------------------------------------------

class TestScalarArithmetic:
    def test_add(self):
        a = np.interval(1.0, 2.0, exact=True)
        b = np.interval(3.0, 4.0, exact=True)
        c = a + b
        assert_contains(c, 4.0, 5.0, 6.0)  # [4,6] rigorously bracketed
        assert c.l <= 4.0 and c.u >= 6.0

    def test_subtract(self):
        a = np.interval(1.0, 2.0, exact=True)
        b = np.interval(3.0, 5.0, exact=True)
        c = a - b
        # [1,2] - [3,5] = [1-5, 2-3] = [-4, -1]
        assert c.l <= -4.0 and c.u >= -1.0

    def test_multiply_positive(self):
        a = np.interval(2.0, 3.0, exact=True)
        b = np.interval(4.0, 5.0, exact=True)
        c = a * b
        assert c.l <= 8.0 and c.u >= 15.0

    def test_multiply_mixed_sign(self):
        a = np.interval(-1.0, 2.0, exact=True)
        b = np.interval(-3.0, 4.0, exact=True)
        c = a * b
        # min = min(3, -4, -6, 8) = -6, max = max(...) = 8
        assert c.l <= -6.0 and c.u >= 8.0

    def test_divide(self):
        a = np.interval(1.0, 2.0, exact=True)
        b = np.interval(2.0, 4.0, exact=True)
        c = a / b
        assert c.l <= 0.25 and c.u >= 1.0

    def test_negate(self):
        a = np.interval(1.0, 2.0, exact=True)
        n = -a
        assert n.l == -2.0 and n.u == -1.0

    def test_power_integer(self):
        a = np.interval(1.0, 2.0, exact=True)
        assert_close_bounds(a ** 2, 1.0, 4.0)
        a2 = np.interval(-1.0, 2.0, exact=True)
        sq = a2 ** 2
        assert sq.l <= 0.0 and sq.u >= 4.0

    def test_power_negative(self):
        a = np.interval(2.0, 4.0, exact=True)
        c = a ** -1
        assert c.l <= 0.25 and c.u >= 0.5

    def test_inverse_positive(self):
        a = np.interval(2.0, 4.0, exact=True)
        inv = a.inverse()
        assert inv.l <= 0.25 and inv.u >= 0.5

    def test_inverse_straddles_zero(self):
        a = np.interval(-1.0, 1.0, exact=True)
        inv = a.inverse()
        assert math.isinf(inv.l) and math.isinf(inv.u)


# ---------------------------------------------------------------------------
# In-place arithmetic
# ---------------------------------------------------------------------------

class TestInplace:
    def test_iadd(self):
        a = np.interval(1.0, 2.0, exact=True)
        a += np.interval(0.5, 1.5, exact=True)
        assert a.l <= 1.5 and a.u >= 3.5

    def test_iadd_scalar(self):
        a = np.interval(1.0, 2.0, exact=True)
        a += 1.0
        assert a.l <= 2.0 and a.u >= 3.0

    def test_imul(self):
        a = np.interval(2.0, 3.0, exact=True)
        a *= np.interval(2.0, 2.0, exact=True)
        assert a.l <= 4.0 and a.u >= 6.0

    def test_isub_scalar(self):
        a = np.interval(5.0, 7.0, exact=True)
        a -= 1.0
        assert a.l <= 4.0 and a.u >= 6.0


# ---------------------------------------------------------------------------
# Scalar mixed with float / int Python objects
# ---------------------------------------------------------------------------

class TestScalarMixed:
    def test_interval_plus_float(self):
        a = np.interval(1.0, 2.0, exact=True)
        c = a + 1.0
        assert c.l <= 2.0 and c.u >= 3.0

    def test_float_plus_interval(self):
        a = np.interval(1.0, 2.0, exact=True)
        c = 1.0 + a
        assert c.l <= 2.0 and c.u >= 3.0

    def test_interval_times_int(self):
        a = np.interval(1.0, 2.0, exact=True)
        c = a * 3
        assert c.l <= 3.0 and c.u >= 6.0

    def test_int_times_interval(self):
        a = np.interval(1.0, 2.0, exact=True)
        c = 3 * a
        assert c.l <= 3.0 and c.u >= 6.0

    def test_scalar_minus_interval(self):
        a = np.interval(1.0, 2.0, exact=True)
        c = 5.0 - a
        # 5 - [1,2] = [3, 4]
        assert c.l <= 3.0 and c.u >= 4.0

    def test_scalar_div_interval(self):
        a = np.interval(2.0, 4.0, exact=True)
        c = 8.0 / a
        # 8 / [2,4] = [2, 4]
        assert c.l <= 2.0 and c.u >= 4.0


# ---------------------------------------------------------------------------
# Rigorous (outward) rounding sanity
# ---------------------------------------------------------------------------

class TestRigorousRounding:
    def test_one_third_strict_containment(self):
        # 1/3 has no exact double representation; the result must contain the
        # true value, with at least one side strictly outward of the nearest
        # double approximation.
        c = 1.0 / np.interval(3.0, 3.0, exact=True)
        nearest = 1.0 / 3.0
        assert c.l <= nearest <= c.u
        assert c.l < nearest or c.u > nearest

    def test_sum_widening_preserves_truth(self):
        a = np.interval(0.1, 0.1, exact=True)
        b = np.interval(0.2, 0.2, exact=True)
        c = a + b
        # 0.1 + 0.2 is not exactly 0.3 in double, but the interval must
        # contain the mathematical value 0.3.
        assert c.l <= 0.3 <= c.u


# ---------------------------------------------------------------------------
# Set operations
# ---------------------------------------------------------------------------

class TestSetOps:
    def test_union(self):
        a = np.interval(1.0, 3.0, exact=True)
        b = np.interval(2.0, 5.0, exact=True)
        u = a.union(b)
        assert u.l == 1.0 and u.u == 5.0

    def test_intersection(self):
        a = np.interval(1.0, 3.0, exact=True)
        b = np.interval(2.0, 5.0, exact=True)
        x = a.intersection(b)
        assert x.l == 2.0 and x.u == 3.0

    def test_intersection_disjoint_is_nan(self):
        a = np.interval(1.0, 2.0, exact=True)
        b = np.interval(3.0, 4.0, exact=True)
        x = a.intersection(b)
        assert math.isnan(x.l) and math.isnan(x.u)

    def test_subseteq(self):
        a = np.interval(1.0, 2.0, exact=True)
        b = np.interval(0.0, 3.0, exact=True)
        assert a.subseteq(b) is True
        assert b.subseteq(a) is False
        assert a.subseteq(a) is True

    def test_supseteq(self):
        a = np.interval(0.0, 3.0, exact=True)
        b = np.interval(1.0, 2.0, exact=True)
        assert a.supseteq(b) is True
        assert b.supseteq(a) is False

    def test_strict_subset(self):
        a = np.interval(1.0, 2.0, exact=True)
        b = np.interval(0.0, 3.0, exact=True)
        assert a.subset(b) is True
        assert a.subset(a) is False

    def test_minimum_maximum(self):
        a = np.interval(1.0, 4.0, exact=True)
        b = np.interval(2.0, 3.0, exact=True)
        mn, mx = a.minimum(b), a.maximum(b)
        assert mn.l == 1.0 and mn.u == 3.0
        assert mx.l == 2.0 and mx.u == 4.0


# ---------------------------------------------------------------------------
# Norm / width
# ---------------------------------------------------------------------------

class TestNorm:
    def test_norm_scalar(self):
        a = np.interval(1.0, 4.0, exact=True)
        assert a.norm() == 3.0

    def test_norm_array_ufunc(self):
        arr = np.array([np.interval(1.0, 2.0, exact=True),
                        np.interval(0.0, 5.0, exact=True)])
        w = np.norm(arr)
        assert np.allclose(w, [1.0, 5.0])


# ---------------------------------------------------------------------------
# Equality / hashing / repr
# ---------------------------------------------------------------------------

class TestEqualityRepr:
    def test_equal_method(self):
        a = np.interval(1.0, 2.0, exact=True)
        b = np.interval(1.0, 2.0, exact=True)
        assert a.equal(b) is True
        assert a.not_equal(b) is False

    def test_not_equal_method(self):
        a = np.interval(1.0, 2.0, exact=True)
        b = np.interval(1.0, 2.5, exact=True)
        assert a.not_equal(b) is True

    def test_array_equal_ufunc(self):
        a = np.array([np.interval(1.0, 2.0, exact=True),
                      np.interval(3.0, 4.0, exact=True)])
        b = np.array([np.interval(1.0, 2.0, exact=True),
                      np.interval(3.0, 5.0, exact=True)])
        eq = a == b
        assert eq.tolist() == [True, False]

    def test_hash_consistent(self):
        a = np.interval(1.0, 2.0, exact=True)
        b = np.interval(1.0, 2.0, exact=True)
        assert hash(a) == hash(b)

    def test_repr_contains_brackets(self):
        s = repr(np.interval(1.0, 2.0, exact=True))
        assert "⟦" in s and "⟧" in s

    def test_zero_and_one_constants(self):
        assert zero.l == 0.0 and zero.u == 0.0
        assert one.l == 1.0 and one.u == 1.0


# ---------------------------------------------------------------------------
# Transcendental ufuncs (correctness, not tightness)
# ---------------------------------------------------------------------------

class TestUFuncs:
    def test_sin_contains_true_values(self):
        a = np.interval(0.0, 1.0, exact=True)
        s = np.sin(a)
        for x in np.linspace(0, 1, 11):
            assert s.l <= math.sin(x) <= s.u

    def test_cos_contains_true_values(self):
        a = np.interval(0.0, 1.0, exact=True)
        c = np.cos(a)
        for x in np.linspace(0, 1, 11):
            assert c.l <= math.cos(x) <= c.u

    def test_sin_wraps_to_full_range(self):
        a = np.interval(0.0, 10.0, exact=True)  # > 2π
        s = np.sin(a)
        assert s.l <= -1.0 and s.u >= 1.0

    def test_exp_monotone(self):
        a = np.interval(0.0, 1.0, exact=True)
        e = np.exp(a)
        assert e.l <= 1.0 <= e.u
        assert e.l <= math.e <= e.u

    def test_sqrt(self):
        a = np.interval(1.0, 4.0, exact=True)
        s = np.sqrt(a)
        assert s.l <= 1.0 and s.u >= 2.0

    def test_arctan(self):
        a = np.interval(0.0, 1.0, exact=True)
        t = np.arctan(a)
        assert t.l <= 0.0 and t.u >= math.pi / 4

    def test_tanh(self):
        a = np.interval(0.0, 1.0, exact=True)
        t = np.tanh(a)
        assert t.l <= 0.0 and t.u >= math.tanh(1.0)

    def test_square_ufunc(self):
        a = np.interval(-2.0, 3.0, exact=True)
        sq = np.square(a)
        assert sq.l <= 0.0 and sq.u >= 9.0


# ---------------------------------------------------------------------------
# Arrays
# ---------------------------------------------------------------------------

class TestArrays:
    def test_array_dtype(self):
        arr = np.array([np.interval(i, i + 1, exact=True) for i in range(4)])
        assert arr.dtype == np.dtype(np.interval)
        assert is_iarray(arr)

    def test_array_add_scalar(self):
        arr = np.array([np.interval(i, i + 1, exact=True) for i in range(3)])
        out = arr + 1.0
        for k in range(3):
            assert out[k].l <= k + 1 and out[k].u >= k + 2

    def test_array_add_interval(self):
        arr = np.array([np.interval(i, i + 1, exact=True) for i in range(3)])
        out = arr + np.interval(0.0, 1.0, exact=True)
        for k in range(3):
            assert out[k].l <= k and out[k].u >= k + 2

    def test_array_ufunc_sin(self):
        arr = np.array([np.interval(0.0, 0.5, exact=True),
                        np.interval(1.0, 1.5, exact=True)])
        s = np.sin(arr)
        for i, lo, hi in [(0, 0.0, 0.5), (1, 1.0, 1.5)]:
            for x in np.linspace(lo, hi, 5):
                assert s[i].l <= math.sin(x) <= s[i].u

    def test_matmul_2x2_vec(self):
        A = np.array([[np.interval(1.0, 2.0, exact=True),
                       np.interval(0.0, 1.0, exact=True)],
                      [np.interval(-1.0, 0.0, exact=True),
                       np.interval(1.0, 3.0, exact=True)]])
        x = np.array([np.interval(1.0, 2.0, exact=True),
                      np.interval(-1.0, 1.0, exact=True)])
        y = A @ x
        # Hand bounds: y[0] in [0,5], y[1] in [-5,3]
        assert y[0].l <= 0.0 and y[0].u >= 5.0
        assert y[1].l <= -5.0 and y[1].u >= 3.0

    def test_matmul_matrix_matrix(self):
        A = np.array([[np.interval(1, 2, exact=True),
                       np.interval(0, 1, exact=True)],
                      [np.interval(-1, 0, exact=True),
                       np.interval(1, 3, exact=True)]])
        I = np.array([[np.interval(1, 1, exact=True),
                       np.interval(0, 0, exact=True)],
                      [np.interval(0, 0, exact=True),
                       np.interval(1, 1, exact=True)]])
        out = A @ I
        for i in range(2):
            for j in range(2):
                assert out[i, j].l <= A[i, j].l <= A[i, j].u <= out[i, j].u

    def test_array_sum(self):
        arr = np.array([np.interval(i, i + 1, exact=True) for i in range(3)])
        s = arr.sum()
        # sum of [0,1]+[1,2]+[2,3] = [3,6]
        assert s.l <= 3.0 and s.u >= 6.0


# ---------------------------------------------------------------------------
# Helper functions in interval/__init__.py
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_get_lu(self):
        l = np.array([0.0, 1.0, 2.0])
        u = np.array([0.5, 1.5, 2.5])
        ix = get_iarray(l, u)
        lo, hi = get_lu(ix)
        # widened by 1 ULP on each side
        assert np.all(lo <= l) and np.all(hi >= u)

    def test_as_lu_shape(self):
        ix = np.array([np.interval(0.0, 1.0, exact=True),
                       np.interval(1.0, 2.0, exact=True)])
        lu = as_lu(ix)
        assert lu.shape == (2, 2)
        assert np.allclose(lu[:, 0], [0.0, 1.0])
        assert np.allclose(lu[:, 1], [1.0, 2.0])

    def test_from_cent_pert_get_cent_pert_roundtrip(self):
        cent = np.array([0.0, 1.0, 2.0])
        pert = np.array([0.1, 0.2, 0.5])
        ix = from_cent_pert(cent, pert)
        c2, p2 = get_cent_pert(ix)
        assert np.allclose(c2, cent, atol=1e-12)
        assert np.allclose(p2, pert, atol=1e-12)

    def test_from_cent_pert_shape_mismatch(self):
        with pytest.raises(Exception):
            from_cent_pert(np.array([1.0]), np.array([0.1, 0.2]))

    def test_has_nan_false(self):
        ix = get_iarray(np.array([0.0, 1.0]), np.array([1.0, 2.0]))
        assert not has_nan(ix)

    def test_has_nan_true(self):
        # intersection of disjoint intervals → NaN
        a = np.interval(1.0, 2.0, exact=True)
        b = np.interval(3.0, 4.0, exact=True)
        bad = a.intersection(b)
        arr = np.array([bad, np.interval(0.0, 1.0, exact=True)])
        assert bool(has_nan(arr))

    def test_get_half_intervals(self):
        # Builds 2**n sub-boxes from one box
        ix = get_iarray(np.array([0.0, 0.0]), np.array([2.0, 4.0]))
        halves = get_half_intervals(ix)
        assert len(halves) == 4  # 2**2


# ---------------------------------------------------------------------------
# Pickling
# ---------------------------------------------------------------------------

class TestPickle:
    def test_scalar_roundtrip(self):
        a = np.interval(1.0, 2.0, exact=True)
        b = pickle.loads(pickle.dumps(a))
        assert a.l == b.l and a.u == b.u
