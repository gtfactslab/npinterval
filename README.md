# npinterval

Interval arithmetic as a NumPy 2 dtype, backed by
[Boost.Interval](https://www.boost.org/doc/libs/release/libs/numeric/interval/)
for rigorous directed rounding.

Intervals print as `⟦lower, upper⟧`.

## Requirements

- Python ≥ 3.8
- NumPy ≥ 2.0
- Boost headers (`apt install libboost-dev`, `brew install boost`, or
  `conda install -c conda-forge boost-cpp`)
- A C++17 compiler with `-frounding-math` (g++, clang)

## Install

```bash
git clone https://github.com/gtfactslab/npinterval.git
cd npinterval
pip install .
```

## Quick start

```python
import numpy as np
import interval                    # registers np.interval
```

### Creating intervals

By default both bounds are widened outward by ±1 ULP so that the interval
provably contains the real number the float approximates. Pass `exact=True`
to use the values as-is:

```python
a = np.interval(1.0, 2.0)              # ⟦1, 2⟧  (bounds widened by ±1 ULP)
p = np.interval(3.0)                   # ⟦3, 3⟧  (point, widened by ±1 ULP)

a_ex = np.interval(1.0, 2.0, exact=True)   # bounds exact
p_ex = np.interval(3.0,       exact=True)  # degenerate point ⟦3, 3⟧
```

### Scalar arithmetic

```python
a = np.interval(1.0, 2.0)
b = np.interval(-1.0, 1.0)

a + b    # ⟦-3.33e-16, 3⟧
a - b    # ⟦-3.33e-16, 3⟧
a * b    # ⟦-2, 2⟧
a ** 2   # ⟦1, 4⟧
-a       # ⟦-2, -1⟧
```

### Rigorous floating-point bounding

Arithmetic uses Boost.Interval's directed-rounding policy:

```python
with np.printoptions(precision=20):
    print(1 / np.interval(3.))
    # ⟦0.33333333333333325932, 0.33333333333333342585⟧
```

### Arrays

```python
arr = np.array([np.interval(float(i), float(i+1)) for i in range(4)])
# [⟦-4.94e-324, 1⟧ ⟦1, 2⟧ ⟦2, 3⟧ ⟦3, 4⟧]

arr * 2
arr + np.interval(-0.5, 0.5)
```

### Matrix–vector multiply

```python
A = np.array([[np.interval(1.0, 2.0), np.interval(0.0, 1.0)],
              [np.interval(-1.0, 0.0), np.interval(1.0, 3.0)]])
x = np.array([np.interval(1.0, 2.0), np.interval(-1.0, 1.0)])

A @ x    # [⟦-8.88e-16, 5⟧  ⟦-5, 3⟧]
```

### NumPy ufuncs

```python
th = np.array([np.interval(0.0, 0.5), np.interval(1.0, 1.5)])

np.sin(th)    # [⟦-1.39e-15, 0.47942554⟧  ⟦0.84147098, 0.99749499⟧]
np.exp(th)    # [⟦1, 1.6487213⟧  ⟦2.7182818, 4.4816891⟧]
np.sqrt(th)   # [⟦0, 0.70710678⟧  ⟦1, 1.2247449⟧]
```

Supported elementwise ufuncs: `add`, `subtract`, `multiply`, `true_divide`,
`floor_divide`, `negative`, `positive`, `square`, `sin`, `cos`, `tan`,
`arctan`, `tanh`, `exp`, `sqrt`, `matmul`, `maximum`, `minimum`,
`equal`, `not_equal`, plus the custom ufuncs `norm`, `union`, `intersection`,
`subseteq`, `supseteq`, `subset`, `supset`.

### Building interval arrays

```python
from interval import from_cent_pert, get_cent_pert, get_lu, get_iarray

l = np.array([-0.1, 0.8, 1.5])
u = np.array([ 0.1, 1.2, 2.5])
ix = get_iarray(l, u)                  # [⟦-0.1, 0.1⟧ ⟦0.8, 1.2⟧ ⟦1.5, 2.5⟧]

ix = from_cent_pert(np.array([0., 1., 2.]),
                    np.array([0.1, 0.2, 0.5]))

lower, upper = get_lu(ix)
c, p          = get_cent_pert(ix)
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

## Tests

```bash
conda install -c conda-forge boost-cpp pytest
pip install -e .
pytest tests/
```
