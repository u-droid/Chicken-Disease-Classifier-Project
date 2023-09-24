import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from cnn_classifier import logger
from cnn_classifier.entity.config_entity import TrainingConfig
from pathlib import Path

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.use_cuda = torch.cuda.is_available()

    def get_base_model(self):
        self.model = torch.load(self.config.updated_base_model_path)
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

    def train(self, train_loader, valid_loader):
        self.get_base_model()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=self.config.params_learning_rate)
        logger.info('Starting Model Training')

        for epoch in range(1, self.config.params_epochs+1):
            train_loss = 0.0
            self.model.train()
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                if self.use_cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # print statistics
                train_loss += loss.item()
                train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
                if batch_idx % 20 == 19:
                    logger.info(f'[{epoch}, {batch_idx + 1:5d}] loss: {train_loss:.3f}')

        logger.info("Finished Training")
        self.save_model(self.config.trained_model_path, self.model)
        logger.info("Saved Trained Model")

    @staticmethod
    def save_model(path: Path, model):
        torch.save(model, path)
