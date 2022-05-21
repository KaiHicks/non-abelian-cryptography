from itertools import permutations
from math import ceil, log2

import numpy as np
from galois import GF
from sympy import Matrix

from util import elem


# elems512_4x128-159
# 512 bit | 24,576 bit public key
d = 2**128-159
n = 4
# 256 bit | 12,288 bit public key
d = 2**64-59
n = 4
# 128 bit | 6,144 bit public key
d = 2**32-17
n = 4

# 320 bit | 32,000 bit public key
d = 2**64-59
n = 5
# 160 bit | 16,000 bit public key
d = 2**32-17
n = 5
# 80 bit | 8,000 bit public key
d = 2**16-17
n = 5


# Get our elementary matrices
emats = elem(n, d)

print(f'Number of elementary matrices: {len(emats)*(d-1)}')
print(f'Number that we have to consider: {len(emats)}')

# Compute the size of the public key
# Number of matrices * elements/matrix * bits/element
pub_size = len(emats)*n*n*ceil(log2(d))
print(f'Public key size: {pub_size/8000:,.3f} kB')
print(f'Public key size: {pub_size//8:,} B')
print(f'Public key size: {pub_size:,} b')

# Compute the log2 of the number of secret keys to get the strength
secret_strength = log2(d**n - d**2)
print(f'Security of secret key: {secret_strength:.3f}')