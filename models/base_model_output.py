class AbstractModelOutput:

    PROPERTIES = NotImplemented

    def __init__(self, **kwargs) -> None:

        if self.PROPERTIES is NotImplemented:
            raise NotImplementedError("ModelOutputs need to have defined a set of Properties")

        for k,v in kwargs.items():
            if k not in self.PROPERTIES:
                raise AttributeError(f"Property {k} not in parameters: {self.PROPERTIES}")
            setattr(self, k, v)

    #! Adding support for dictionary converstion and ** dereferencing
    def keys(self):
        return_keys = []
        for key in self.PROPERTIES:
            if self[key] is not None:
                return_keys.append(key)
        return return_keys

    def __getitem__(self, key):
        if key not in self.PROPERTIES:
            raise KeyError(f"The parameter {key} is not set for class {self.__class__.__name__}")
        return getattr(self, key, None)
    
    def __setitem__(self, key, value):
        if key not in self.PROPERTIES:
            raise KeyError(f"The parameter {key} is not in the class {self.__class__.__name__} PARAMETERS List")
        setattr(self, key, value)

    
    def __contains__(self, key):
        return key in self.keys()