
from abc import abstractmethod

class _GenerationMixin:
    @abstractmethod
    def generate():
        """To generate something.
        """
    
    @abstractmethod
    def generate_stream():
        """To generate something in a stream.
        """