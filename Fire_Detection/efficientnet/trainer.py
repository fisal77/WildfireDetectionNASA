import os
import shutil
from abc import ABCMeta, abstractmethod

import mlconfig
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils import data
from tqdm import tqdm, trange

from .metrics import Average#, Accuracy


class AbstractTrainer(metaclass=ABCMeta):

    @abstractmethod
    def fit(self):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError


@mlconfig.register
class Trainer(AbstractTrainer):

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, train_loader: data.DataLoader,
                 valid_loader: data.DataLoader, scheduler: optim.lr_scheduler._LRScheduler, device: torch.device,
                 num_epochs: int, batch_size: int, output_dir: str):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.num_epochs = num_epochs
        self.output_dir = output_dir

        self.epoch = 1
        self.best_loss = 1000

    def fit(self):
        epochs = trange(self.epoch, self.num_epochs + 1, desc='Epoch', ncols=0)
        for self.epoch in epochs:
            self.scheduler.step()

            train_loss = self.train()
            valid_loss = self.evaluate()

            self.save_checkpoint(os.path.join(self.output_dir, 'checkpoint.pth'))
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss.value
                self.save_checkpoint(os.path.join(self.output_dir, 'best.pth'))

            epochs.set_postfix_str(f'train loss: {train_loss}, '
                                   f'valid loss: {valid_loss}, '
                                   f'best valid loss: {self.best_loss:.2f}')

    def train(self):
        self.model.train()
        criterion = nn.BCEWithLogitsLoss()

        train_loss = Average()
        #train_acc = Accuracy()
        train_loader = tqdm(data.DataLoader(self.train_loader, batch_size=self.batch_size, shuffle=True, num_workers=28), ncols=0, desc="Train")
        #train_loader = self.train_loader(DataLoader(train=True, batch_size=))
        for x, y in train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            output = self.model(x)
            loss = criterion(output, torch.reshape(y, (y.shape[0], 1)).float())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss.update(loss.item(), number=x.size(0))
            #train_acc.update(output, y)

            train_loader.set_postfix_str(f'train loss: {train_loss}.')

        return train_loss

    def evaluate(self):
        self.model.eval()

        valid_loss = Average()
        #valid_acc = Accuracy()
        criterion = nn.BCEWithLogitsLoss()
        valid_loader = tqdm(data.DataLoader(self.valid_loader, batch_size=self.batch_size, shuffle=False, num_workers=28), desc="Validate", ncols=0) 
        with torch.no_grad():
            #valid_loader = self.valid_loader(DataLoader())
            for x, y in valid_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)
                loss = criterion(output, torch.reshape(y, (y.shape[0], 1)).float())

                valid_loss.update(loss.item(), number=x.size(0))
                #valid_acc.update(output, y)

                valid_loader.set_postfix_str(f'valid loss: {valid_loss}.')

        return valid_loss

    def save_checkpoint(self, f):
        self.model.eval()

        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'best_loss': self.best_loss
        }

        dirname = os.path.dirname(f)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        torch.save(checkpoint, f)

    def resume(self, f):
        checkpoint = torch.load(f, map_location=self.device)

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

        self.epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['best_loss']
