import torch
import torchvision
import torch.nn as nn
from pathlib import Path
from cnn_classifier import logger
from cnn_classifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        self.model = torchvision.models.vgg16(pretrained=True)
        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, num_classes, learning_rate):
        # Freeze the parameters of the model.
        for param in model.parameters():
            param.requires_grad = False

        n_inputs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(n_inputs, num_classes)
        # check if CUDA is available
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            model = model.cuda()
        return model

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            num_classes=self.config.params_classes,
            learning_rate=self.config.params_learning_rate
        )
        logger.info(self.full_model)
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model):
        torch.save(model.state_dict(), path)
