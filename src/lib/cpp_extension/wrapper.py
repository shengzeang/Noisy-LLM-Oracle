import os
from torch.utils.cpp_extension import load

dir_path = os.path.dirname(os.path.realpath(__file__))

utils = load(name='utils', sources=[os.path.join(dir_path, 'utils.cpp')], extra_cflags=['-fopenmp', '-O3'], extra_ldflags=['-lgomp','-lrt'])
