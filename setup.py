from setuptools import setup, Extension
import numpy as np

if __name__ == '__main__':
    setup(
        ext_modules=[
            Extension(
                name='interval.numpy_interval',
                sources=[
                    'interval/numpy_interval.cpp',
                ],
                depends=[
                    'interval/interval.hpp',
                    'interval/numpy_interval.cpp',
                ],
                include_dirs=[
                    np.get_include(),
                    'interval',
                ],
                language='c++',
                extra_compile_args=[
                    '-std=c++17',
                    '-frounding-math',   # required for Boost.Interval directed rounding
                ],
            )
        ]
    )
