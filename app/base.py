from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def prepare(self):
        pass

    @abstractmethod
    def createAndTrain(self, train_data, test_data, saved_model_path):
        pass

    @abstractmethod
    def test(self, model, test_data):
        pass

    @abstractmethod
    def future(self, model, days):
        pass
