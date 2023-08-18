
__all__ = [
    'interval', 'zero', 'one'
]

import numpy
from npinterval.interval.numpy_interval import *

if numpy.__dict__.get('interval') is not None:
    raise RuntimeError('The NumPy package already has an interval type')

numpy.interval = interval
numpy.sctypeDict['interval'] = numpy.dtype(interval)

# numba.typeDict = numba.from_dtype(numpy.interval)


def as_lu (iarray) :
    iarray = numpy.asarray(iarray)
    return numpy.array([i.vec for i in iarray.reshape(-1)]).reshape(iarray.shape + (2,))

def get_lu (iarray) :
    iarray = numpy.asarray(iarray)
    shape = iarray.shape
    lu = numpy.array([i.vec for i in iarray.reshape(-1)])
    return lu[:,0].reshape(shape), lu[:,1].reshape(shape)

def as_iarray (lu) :
    lu = numpy.asarray(lu)
    return numpy.array([(numpy.interval(lui[0],lui[1]) if lui[0] < lui[1] else numpy.interval(lui[1],lui[0])) \
                        for lui in lu.reshape(-1,2)]).reshape((lu.shape[0],lu.shape[1]))

def get_iarray (l, u) :
    shape = l.shape
    return as_iarray(numpy.dstack((l, u))).reshape(shape)

is_iarray = lambda x : x.dtype == numpy.interval

def from_cent_pert (cent, pert) :
    cent = numpy.asarray(cent)
    pert = numpy.asarray(pert)
    shape = cent.shape
    if cent.shape != pert.shape :
        raise Exception("cent and pert shapes should match")
    # ret = numpy.empty_like(cent).reshape(-1)
    cent = cent.reshape(-1)
    pert = pert.reshape(-1)
    return numpy.array([numpy.interval(cent[i]-pert[i],cent[i]+pert[i]) for i in range(len(cent))]).reshape(shape)

def get_cent_pert (iarray) :
    l, u = get_lu(iarray)
    return (l + u)/2, (u - l)/2

def width (iarray, scale=None) :
    width = numpy.norm(iarray)
    return width if scale is None else width / scale

def get_half_intervals (iarray) :
    _x, x_ = get_lu(iarray)
    part_avg = (_x + x_) / 2
    _xx_ = numpy.concatenate((_x,x_))
    n = len(_x)
    ret = []
    for part_i in range(2**n) :
        part = numpy.copy(_xx_)
        for ind in range (n) :
            part[ind + n*((part_i >> ind) % 2)] = part_avg[ind]
        ret.append(get_iarray(part[:n], part[n:]))
    return ret

# import shapely.geometry
# sg_box = lambda x, xi=0, yi=1 : shapely.geometry.box(x[xi].l,x[yi].l,x[xi].u,x[yi].u)

from math import isnan
def has_nan (iarray) :
    l,u = get_lu(iarray)
    return numpy.any(numpy.isnan(l)) or numpy.any(numpy.isnan(u))

zero = numpy.interval(0,0)
one = numpy.interval(1,1)
