# npinterval

Interval arithmetic as a NumPy 2 dtype, backed by
[Boost.Interval](https://www.boost.org/doc/libs/release/libs/numeric/interval/)
for rigorous directed rounding.

Intervals are printed as `⟦lower, upper⟧`.

## Requirements

- Python ≥ 3.8
- NumPy ≥ 2.0
- Boost headers (`libboost-dev` on Debian/Ubuntu, `brew install boost` on macOS, or `conda install -c conda-forge boost-cpp`)

## Installation

```bash
git clone https://github.com/gtfactslab/npinterval.git
cd npinterval
pip install .
```

## Basic usage

```python
import numpy as np
import interval                 # registers np.interval
```

### Creating intervals

By default both bounds are widened outward by ±1 ULP, guaranteeing that the interval
contains the real number the float approximates. Pass `exact=True` to use the values as-is:

```python
a = np.interval(1.0, 2.0)             # ⟦0.99999999999999989, 2.0000000000000004⟧
b = np.interval(-1.0, 1.0)            # ⟦-1.0000000000000002, 1.0000000000000002⟧
p = np.interval(3.0)                  # ⟦2.9999999999999996, 3.0000000000000004⟧

# exact=True: treat float values as exact bounds (no widening)
a_ex = np.interval(1.0, 2.0, exact=True)   # ⟦1, 2⟧
p_ex = np.interval(3.0, exact=True)        # ⟦3, 3⟧  (degenerate / point interval)
```

### Scalar arithmetic

```python
a + b    # ⟦-3.3306691e-16, 3⟧
a - b    # ⟦-3.3306691e-16, 3⟧
a * b    # ⟦-2, 2⟧
a ** 2   # ⟦1, 4⟧
-a       # ⟦-2, -1⟧
```

### Rigorous floating-point bounding

Arithmetic results are rigorously bounded using directed rounding via
Boost.Interval and `-frounding-math`.

```python
with np.printoptions(precision=20):
    print(1 / np.interval(3.))
    # ⟦0.33333333333333325932, 0.33333333333333342585⟧
```

### Arrays of intervals

```python
# Build an interval array element-by-element
arr = np.array([np.interval(float(i), float(i+1)) for i in range(4)])
# [⟦-4.9406565e-324, 1⟧  ⟦1, 2⟧  ⟦2, 3⟧  ⟦3, 4⟧]

arr * 2
# [⟦-9.8813129e-324, 2⟧  ⟦2, 4⟧  ⟦4, 6⟧  ⟦6, 8⟧]

arr + np.interval(-0.5, 0.5)
# [⟦-0.5, 1.5⟧  ⟦0.5, 2.5⟧  ⟦1.5, 3.5⟧  ⟦2.5, 4.5⟧]
```

### Matrix–vector multiply

```python
A = np.array([[np.interval(1.0, 2.0), np.interval(0.0, 1.0)],
              [np.interval(-1.0, 0.0), np.interval(1.0, 3.0)]])

x = np.array([np.interval(1.0, 2.0), np.interval(-1.0, 1.0)])

A @ x
# [⟦-8.8817842e-16, 5⟧  ⟦-5, 3⟧]
```

### NumPy ufuncs

Standard NumPy math functions work directly on interval arrays:

```python
th = np.array([np.interval(0.0, 0.5), np.interval(1.0, 1.5)])

np.sin(th)    # [⟦-1.3935e-15, 0.47942554⟧  ⟦0.84147098, 0.99749499⟧]
np.exp(th)    # [⟦1, 1.6487213⟧  ⟦2.7182818, 4.4816891⟧]
np.sqrt(th)   # [⟦0, 0.70710678⟧  ⟦1, 1.2247449⟧]
```
Supported elementwise ufuncs include: `add`, `subtract`, `multiply`, `true_divide`,
`floor_divide`, `negative`, `positive`, `square`, `sin`, `cos`, `tan`,
`arctan`, `tanh`, `exp`, `sqrt`, `matmul`, `maximum`, `minimum`,
`equal`, `not_equal`, plus the custom ufuncs `norm`, `union`, `intersection`,
`subseteq`, `supseteq`, `subset`, `supset`.

### Constructing np.Arrays of interval dtype

```python
from interval import from_cent_pert, get_cent_pert, get_lu, get_iarray

# From lower and upper bound arrays directly
l = np.array([-0.1, 0.8, 1.5])
u = np.array([ 0.1, 1.2, 2.5])
ix = get_iarray(l, u)
# [⟦-0.1, 0.1⟧  ⟦0.8, 1.2⟧  ⟦1.5, 2.5⟧]

# From center and perturbation
center       = np.array([0.0, 1.0, 2.0])
perturbation = np.array([0.1, 0.2, 0.5])
ix = from_cent_pert(center, perturbation)
# [⟦-0.1, 0.1⟧  ⟦0.8, 1.2⟧  ⟦1.5, 2.5⟧]

lower, upper = get_lu(ix)
# lower: [-0.1  0.8  1.5]
# upper: [ 0.1  1.2  2.5]

c, p = get_cent_pert(ix)
# c: [0. 1. 2.]   p: [0.1 0.2 0.5]
```

### Set operations

```python
a = np.interval(1.0, 3.0)
b = np.interval(2.0, 5.0)

a.union(b)          # ⟦1, 5⟧
a.intersection(b)   # ⟦2, 3⟧
a.subseteq(b)       # False
```

Disjoint intersections return `⟦nan, nan⟧`.

## Notes on equality

`a == b` on bare interval scalars falls back to Python's identity comparison
(intervals do not define `tp_richcompare`). Use `a.equal(b)`, `a.not_equal(b)`,
or wrap in arrays — the ufuncs compare bounds elementwise:

```python
np.array([a]) == np.array([b])        # uses the equal ufunc
```
