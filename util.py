
from functools import cache
from itertools import permutations

import numpy as np
from galois import GF

class ElementaryMatrix:
    def __init__(self, r1, r2, power=1):
        self.r1 = r1
        self.r2 = r2
        self.power = power
    
    def __pow__(self, power):
        return ElementaryMatrix(self.r1, self.r2, self.power*power)

    def __matmul__(self, other):
        cpy = other.copy()
        cpy[self.r1, :] += self.power*cpy[self.r2, :]
        return cpy


@cache
def elem(n, d):
    '''
    Returns a list of all additive elementary matrices over F_d with no scalar
    multiplier
    '''
    F = GF(d)
    emats = []
    # Iterate over all pairs of (distinct) rows
    for r1, r2 in permutations(range(n), 2):
        # Get the identity and then apply the row operation
        emats.append(ElementaryMatrix(r1, r2))
    
    return emats

@cache
def elem_indices(n, d):
    ...

def elem_decomp(A, d):
    '''
    Returns a list [(E_1, p_1), (E_2, p_2), ...] where  
    E_1^{p_1} E_2^{p_2}... = A over F_d and each E_i is elementary .
    '''
    n = A.shape[0]
    emats = elem(n, d)


if __name__ == '__main__':
    n = 4
    d = 2**16 - 17
    F = GF(5)

    E = F(np.array([
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]))
    
    print(E @ E)

    print('from elem')

    elems = elem(n, 5)

    e1 = ElementaryMatrix(1, 2)
    e2 = ElementaryMatrix(2, 1)
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert (e2.__rmatmul__(A.T).T == e1.__matmul__(A)).all(), \
        f'{e2.__rmatmul__(A.T).T}\n{e1.__matmul__(A)}'