from setuptools import setup, Extension
import numpy as np

if __name__ == '__main__' :
    setup(
        ext_modules=[
            Extension(
                name='npinterval.interval.numpy_interval',
                sources=[
                    'interval/interval.c',
                    'interval/numpy_interval.c'
                ],
                depends=[
                    "interval/interval.h",
                    'interval/interval.c',
                    'interval/numpy_interval.c'
                ],
                include_dirs=[
                    np.get_include(),
                    "interval"
                ]
            )
        ]
    )

# from distutils.core import setup, Extension
# import numpy as np

# ext_modules = []

# ext = Extension('interval.numpy_interval',
#                 sources=['interval/interval.c',
#                          'interval/numpy_interval.c'],
#                 include_dirs=[np.get_include()],
#                 extra_compile_args=['-std=c99'])
# ext_modules.append(ext)

# setup(name='npinterval',
#       version='0.1',
#       description='Interval NumPy type extension',
#     #   packages=['npinterval', 'npinterval.interval'],
#       packages=['interval'],
#       ext_modules=ext_moduleps)
