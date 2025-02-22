import time
from code.function import *
from tqdm import tqdm
import matplotlib.pyplot as plt


class ModelTrainer:
    def __init__(self,
                 model,
                 optimizer,
                 device,
                 name,
                 early_stop=5,
                 n_epochs=100,
                 seed=0):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.early_stop = early_stop
        self.n_epochs = n_epochs
        self.seed = seed
        self.name = name

        self.model.to(self.device)

    def _train_epoch(self, train_dl, epoch):
        ic = 0
        self.model.train()
        tqdm_ = tqdm(iterable=train_dl)
        i = 0
        for i, batch in enumerate(tqdm_):
            x1, labels = batch
            x1 = x1.to(self.device)
            labels = labels.to(self.device)
            out, _ = self.model(x1)

            loss = pearson_r_loss(labels, out)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            ic_i = pearson_r(labels, out).item()
            ic += ic_i
            tqdm_.set_description("epoch:{:d} train IC:{:.4f}".format(epoch, ic / (i + 1)))
        return ic / (i + 1)

    def _eval_epoch(self, val_dl, epoch):
        self.model.eval()
        ic = 0
        tqdm_ = tqdm(iterable=val_dl)
        i = 0
        for i, batch in enumerate(tqdm_):
            x1, labels = batch
            x1 = x1.to(self.device)
            labels = labels.to(self.device)
            out, _ = self.model(x1)

            ic += pearson_r(labels, out).item()

            tqdm_.set_description(
                "epoch:{:d} test IC:{:.4f} ".format(epoch, ic / (i + 1)))
        return ic / (i + 1)

    def fit(self, train_dl, test_dl, model_path):
        print(f'current device: {self.device}')
        print(f'begin time: {time.ctime()}')
        print(self.model)
        set_seed(self.seed)

        max_ic = -10000
        max_epoch = 0

        train_list = []
        val_list = []

        epoch = 0
        for epoch in range(self.n_epochs):

            train_ic = self._train_epoch(train_dl, epoch)
            ic = self._eval_epoch(test_dl, epoch)

            train_list.append(train_ic)
            val_list.append(ic)

            if ic > max_ic:
                max_ic = ic
                max_epoch = epoch
                torch.save(self.model, f'{model_path}/{self.name}.pt')
            else:
                if epoch - max_epoch >= self.early_stop:
                    break

        fig = plt.figure(figsize=[8, 6])
        plt.plot(
            np.arange(epoch + 1),
            train_list,
            label='train_scores'
        )
        plt.plot(
            np.arange(epoch + 1),
            val_list,
            label='valid_scores'
        )
        plt.legend()
        plt.title(f"scores for {self.name} best IC = {max_ic}")
        fig.savefig(f'{model_path}/{self.name}-loss.png')

        return train_list, val_list

    def predict(self, val_dl):
        x1, labels = next(iter(val_dl))
        x1 = x1.to(self.device)
        self.model.eval()
        y_pred, _ = self.model(x1)
        y_pred = y_pred.cpu().detach().numpy()

        return y_pred
