from __future__ import annotations
from base64 import b85decode, b85encode

from itertools import product
from json import dumps
from multiprocessing import Pool, cpu_count
from secrets import randbelow
from random import randrange
from typing import List

import numpy as np
from galois import GF, FieldArray
from simplejson import loads
from yaml import serialize

from keypair import KeyPair, PublicKey, SecretKey
from util import elem

class SimMatKeyPair(KeyPair):
    @staticmethod
    def generate(n:int, d:int):
        skey = SimMatSecretKey.generate(n, d)
        pkey = SimMatPublicKey.from_secret(skey)

        return SimMatKeyPair(skey, pkey)

    def __init__(self, secret_key:SimMatSecretKey, public_key:SimMatPublicKey):
        self._secret_key = secret_key
        self._public_key = public_key
    
    @property
    def public_key(self) -> PublicKey:
        return self._public_key

class SimMatSecretKey(SecretKey):
    @staticmethod
    def generate(n:int, d:int, elems:List[FieldArray]=None)->SimMatSecretKey:
        elems = elems or elem(n, d)

        matrix = np.zeros((n, n), dtype=np.uint)
        # Loop while our matrix is not invertible
        while np.linalg.matrix_rank(matrix) < n:
            # Generate the secret matrix
            for i, j in product(range(n), range(n)):
                matrix[i, j] = randbelow(d)
        
        field = GF(d)
        return SimMatSecretKey(field(matrix))

    def __init__(self, key_mat:FieldArray):
        self.n = key_mat.shape[0]
        self.d = key_mat._order

        self._secret = key_mat
        self._secret_inv = np.linalg.inv(key_mat)
    
    def __call__(self, A:FieldArray, inv:bool=False):
        if not inv:
            k, k_inv = self._secret, self._secret_inv
        else:
            k, k_inv = self._secret_inv, self._secret
        
        # Need to do the right side first since, if A is an ElementaryMatrix
        # then we cant be calling k_inv.__matmul__ first.
        return k_inv @ (A @ k)

class SimMatPublicKey(PublicKey):
    @staticmethod
    def deserialize(dump:str)->SimMatPublicKey:
        payload = loads(dump)
        n = payload['n']
        d = payload['d']
        dtype = np.dtype(payload['dtype'])
        mat_strs = payload['mats']

        mats = []
        field = GF(d)

        for mat_str in mat_strs:
            mat_bytes = b85decode(mat_str)
            matrix = np.frombuffer(mat_bytes, dtype=dtype)
            matrix = np.reshape(matrix, (n, n))
            mats.append(field(matrix))
        
        elems = elem(n, d)
        return SimMatPublicKey(mats, elems)

    @staticmethod
    def from_secret(
        secret_key:SimMatSecretKey, elems:List[FieldArray]=None
    )->SimMatPublicKey:
        elems = elems or elem(secret_key.n, secret_key.d)
        mats = []
        for emat in elems:
            mats.append(secret_key(emat))
        
        return SimMatPublicKey(mats, elems)
    
    def __init__(self, mats:List[FieldArray], elems:List[FieldArray]=None):
        self._mats = mats
        self.elems = elems or elem(mats[0].shape[0], mats[0]._order)

        self.n = mats[0].shape[0]
        self.d = mats[0]._order
    
    def __eq__(self, other):
        if not isinstance(other, SimMatPublicKey):
            return super().__eq__(other)
        
        for mat1, mat2 in zip(self._mats, other._mats):
            if (mat1 != mat2).any():
                return False
        return True
    
    def key_agree(self, rounds=-1):
        if rounds <= 0:
            rounds = self.n
        field = GF(self.d)

        ephemeral = field(np.eye(self.n, dtype=np.uint, order='F'))
        agreed_key = field(np.eye(self.n, dtype=np.uint, order='F'))
        for _ in range(rounds):
            # Shuffle the indices, cannot use random.shuffle because we *need* this
            # to be secure
            indices = np.arange(len(self._mats))
            # Cannot use enumerate(...) since we will be modifying indices as we go
            for i in range(len(indices)):
                j = randbelow(len(indices) - i) + i
                indices[i], indices[j] = indices[j], indices[i]

            for i in indices:
                j = randbelow(self.d-1) + 1

                # This is not timing safe, but luckily, it doesn't have to be!
                # Every time we index into memory 
                ephemeral = np.linalg.matrix_power(self._mats[i], j) @ ephemeral
                agreed_key = (self.elems[i]**j)@agreed_key

        return ephemeral, agreed_key
    
    def multiprocess_key_agree(self, n_iters:int=-1, cores=None):
        if not cores:
            cores = cpu_count()
        if n_iters <= 0:
            n_iters = self.n**4
        
        def np_key_agree(n_iters:int):
            return np.array(self.key_agree(n_iters=n_iters))
        
        iters_per = [n_iters//cores]*cores
        iters_per[0] += n_iters%cores

        with Pool(cores) as p:
            agreements = p.map(np_key_agree, iters_per)
        
        # Assemble agreements
        field = GF(self.d)
        ephemeral = field(np.eye(self.n))
        agreed_key = field(np.eye(self.n))
        for e, k in agreements:
            ephemeral = ephemeral @ e
            agreed_key = agreed_key @ k
        
        return ephemeral, agreed_key
    
    def serialize(self):
        payload = {
            'n': self.n,
            'd': self.d,
            'dtype': 'uint32',
            'mats': []
        }

        for mat in self._mats:
            mat = np.array(mat, dtype=np.uint32)
            mat_bytes = mat.tobytes('C')
            mat_str = b85encode(mat_bytes).decode('ascii')
            payload['mats'].append(mat_str)
        
        return dumps(payload)

if __name__ == '__main__':
    from tqdm import tqdm
    from time import perf_counter

    skey = SimMatSecretKey.generate(4, 2**32-17)
    pkey = SimMatPublicKey.from_secret(skey)
    print(len(pkey._mats), 'matrices in public key')

    serialized = pkey.serialize()
    assert SimMatPublicKey.deserialize(serialized) == pkey, \
        'Serialization-deserialization failed!'
    print(f'Serialized length: {len(serialized):,} bytes')

    # Key agreement
    print('Starting key agreement')
    trials = 32
    start_time = perf_counter()
    for _ in tqdm(range(trials)):
        e_prime, k_prime = pkey.key_agree()
    total_time = perf_counter() - start_time
    print(f'Agerage time: {total_time/trials:,.3f}ms/iter')
    print(k_prime.flags)
    print('Ephemeral:')
    print(e_prime)
    print('Shared secret:')
    print(k_prime)

    print()
    print('Reassembling from private key')
    k = skey(e_prime, inv=True)
    print('Reconstructed secret:')
    print(k)

    assert (k_prime == k).all(), f'Failed key agreement\n{k_prime - k}'

    # Signature
