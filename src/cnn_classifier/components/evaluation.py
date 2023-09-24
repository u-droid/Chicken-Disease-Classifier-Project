import torch
from pathlib import Path
from urllib.parse import urlparse
from cnn_classifier import logger
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from cnn_classifier.utils.common import read_yaml, create_directories, save_json
from cnn_classifier.entity.config_entity import EvaluationConfig

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.use_cuda = torch.cuda.is_available()

    def get_base_model(self, path):
        self.model = torch.load(path)
        if self.use_cuda:
            self.model = self.model.cuda()

    def get_data_loaders(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        data = ImageFolder(self.config.training_data, transform=transform)
        test_ratio = 0.2
        total_size = len(data)
        test_size = int(test_ratio * total_size)
        train_size = total_size - test_size
        batch_size = self.config.params_batch_size
        # Split the dataset into training and testing
        train_dataset, test_dataset = random_split(data, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        return train_loader, test_loader

    def evaluation(self):
        self.get_base_model(self.config.path_of_model)
        _, valid_loader = self.get_data_loaders()
        criterion = nn.CrossEntropyLoss()
        # Validation
        self.model.eval()
        correct = 0
        total = 0
        valid_loss = 0
        for batch_idx, (inputs, labels) in enumerate(valid_loader):
            if self.use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            with torch.no_grad():
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            logger.info(f"Validation Loss= {valid_loss:.3f} Validation Accuracy = {accuracy:.2f}%")
        score = {"loss": float(valid_loss), "accuracy": accuracy}
        save_json(path=Path("scores.json"), data=score)
        logger.info('Saved Scores')
