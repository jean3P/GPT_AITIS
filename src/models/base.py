# src/models/base.py

from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseModelClient(ABC):
    @abstractmethod
    def query(self, question: str, context_files: List[str]) -> Dict[str, Any]:
        pass

