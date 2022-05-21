from __future__ import annotations

from abc import ABC, abstractproperty, abstractstaticmethod


class KeyPair(ABC):
    @abstractproperty
    def public_key(self)->PublicKey:
        ...

class SecretKey(ABC):
    ...

class PublicKey(ABC):
    @abstractstaticmethod
    def deserialize(dump:str)->PublicKey:
        ...
    
    @abstractproperty
    def serialize(self)->str:
        ...
