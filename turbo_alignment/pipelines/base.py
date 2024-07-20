from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    @abstractmethod
    def run(self, experiment_settings) -> None:
        pass
