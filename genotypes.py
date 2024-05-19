from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat')

PRIMITIVES = [
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

PRIMITIVES_POOL = [
    'max_pool_3x3',
    'avg_pool_3x3']


DARTS = Genotype(normal=[('none', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 2), ('none', 2), ('none', 1), ('none', 4), ('skip_connect', 1), ('max_pool_3x3', 0)], normal_concat=[2,3,4,5])

