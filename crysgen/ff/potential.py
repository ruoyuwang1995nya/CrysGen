from abc import ABC, abstractmethod
from typing import Optional,Union
from pathlib import Path 

class BasePotential(ABC):
    """
    The base class for MLFF model
    """
    __ModelTypes = {}
    
    def __init__(self):
        pass    

    
    @staticmethod
    def register(key:str):
        """
        Register a model type.
        """
        def decorator(cls):
            BasePotential.__ModelTypes[key] = cls
            return cls
        return decorator
    
    @staticmethod
    def get_model(key:str):
        """
        Get a model type by key.
        """
        try:
            return BasePotential.__ModelTypes[key]
        except KeyError:
            raise ValueError(f"Model type '{key}' is not registered.")
        
    @staticmethod
    def get_models():
        """
        Get all registered model types.
        """
        return BasePotential.__ModelTypes.keys()
    
    
    @abstractmethod
    def relax():
        """
        Relax the structure using the potential.
        """
        pass
    @abstractmethod
    def inference():
        """
        Inference the structure using the potential.
        """
        pass