from abc import ABC, abstractmethod
from typing import Optional,Union
from pathlib import Path

class BaseModel(ABC):
    """
    The base class for model training and generation.
    """
    
    __ModelTypes = {}
    
    def __init__(self, model:Union[Path,str]=None ):
        self._model = model
        self._data_train = None
        self._data_val = None
        self._data_test = None
        self._stru_gen = None
        self._stru_gen_traj = None
    
    @property
    def model(self):
        return self._model
    
    @property
    def data_train(self):
        return self._data_train
    
    @property
    def data_val(self):
        return self._data_val
    
    @property
    def data_test(self):
        return self._data_test
    
    @property
    def stru_gen(self):
        return self._stru_gen
    @property
    def stru_gen_traj(self):
        return self._stru_gen_traj
    
    @staticmethod
    def register(key:str):
        """
        Register a model type.
        """
        def decorator(cls):
            BaseModel.__ModelTypes[key] = cls
            return cls
        return decorator
    
    @staticmethod
    def get_model(key:str):
        """
        Get a model type by key.
        """
        try:
            return BaseModel.__ModelTypes[key]
        except KeyError:
            raise ValueError(f"Model type '{key}' is not registered.")
        
    @staticmethod
    def get_models():
        """
        Get all registered model types.
        """
        return BaseModel.__ModelTypes.keys()
    
    @abstractmethod
    def get_data(config:Optional[dict]=None):
        """
        Convert a dictionary to MatterGen arguments.
        """
        pass
    
    @abstractmethod
    def train(self, *args, **kwargs):
        """
        Train the model.
        """
        pass
    @abstractmethod
    def generate(self, *args, **kwargs):
        """
        Generate new samples.
        """
        pass
    
    