import pickle
import sys
import warnings
import random

from tqdm import tqdm

warnings.filterwarnings('ignore')
print("CAREFUL: WARNINGS ARE SUPPRESSED")

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from epochs import TrainEpoch, ValidEpoch
from hflayers import HopfieldLayer, Hopfield
from meter import Precision, Accuracy, Recall
from mnist_c import load_mnistc

data_dir = "data"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

transform_target = transforms.Compose([
    lambda x: torch.tensor(x),
    # lambda x: F.one_hot(x, len(labels).float(),
])

train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=transform,
                                           target_transform=transform_target)
test_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=transform,
                                          target_transform=transform_target)

m = len(train_dataset)

train_data, val_data = random_split(train_dataset, [int(m - m * 0.2), int(m * 0.2)])
batch_size = 256

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def get_corrput_loader(dirname=None):
    return DataLoader(load_mnistc(dirname=dirname, transform=transform), batch_size=batch_size, shuffle=False)


def plot_two_imgs_t(x, x2, labels=None):
    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(x.detach().numpy())
    ax[1].imshow(x2.detach().numpy())
    if labels is not None:
        title = f"PR: {labels['y_pr']}, GT: {labels['y_gt']}"
        ax[1].set_title(title)
    plt.show()


class Encoder(nn.Module):

    def __init__(self,
                 num_input_channels: int,
                 base_channel_size: int,
                 latent_dim: int,
                 act_fn: object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 28x28 => 14x14
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=2, stride=2),  # 14x14 => 8x8
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 8x8 => 4x4
            act_fn(),
            nn.Flatten(),  # Image grid to single feature vector
            nn.Linear(2 * 16 * c_hid, latent_dim)
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):

    def __init__(self,
                 num_input_channels: int,
                 base_channel_size: int,
                 latent_dim: int,
                 act_fn: object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2 * 16 * c_hid),
            act_fn()
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2 * c_hid, 2 * c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=2, stride=2),  # 8x8 => 14x14
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2),
            # 16x16 => 32x32
            nn.Tanh()  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x


class Autoencoder(nn.Module):

    def __init__(self,
                 base_channel_size: int,
                 latent_dim: int,
                 encoder_class: object = Encoder,
                 decoder_class: object = Decoder,
                 num_input_channels: int = 1,
                 width: int = 32,
                 height: int = 32):
        super().__init__()
        self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim)
        self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim)

    def forward(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


class CustomHopfield(Hopfield):

    def __init__(self, embeddings, *args, **kwargs):
        super(CustomHopfield, self).__init__(*args, **kwargs)
        self.keys = embeddings[..., :-1].repeat(256, 1, 1)  # torch.transpose(embeddings[..., :-1], 0, 1).repeat(256, 1, 1)
        self.beta = torch.tensor(2.).repeat(256, 1).unsqueeze(2)

    def forward(self, query):
        return super(CustomHopfield, self).forward(self.beta, query, self.keys)


class HopfieldModel(nn.Module):

    def __init__(self, encoder, embeddings, n_networks=4, enc_emb_size=128):
        super(HopfieldModel, self).__init__()
        self.encoder = encoder
        # TODO: actual image must be query, all vectores must be stored as keys beforehand, or calculate average embedding per class and then query it?
        self.hopfield_networks = self.get_hopfield_networks(n_networks, embeddings)
        self.linear = nn.Linear(enc_emb_size*n_networks, 10)

    def forward(self, x, return_rec_img=False, with_hopfield=True):
        z = self.encoder(x)
        if with_hopfield:
            z2 = []
            for hn in self.hopfield_networks:
                z2.append(hn(z.unsqueeze(1)).squeeze())
            z = torch.concat(z2)
        out = F.log_softmax(self.linear(F.relu(z)), dim=1)

        if return_rec_img:
            return out, z
        else:
            return out

    def get_hopfield_networks(self, n_networks, embeddings):
        assert embeddings.shape[0] % n_networks == 0
        n_embeddings = int(embeddings.shape[0] / n_networks)

        networks = []
        for i in range(n_networks):
            networks.append(CustomHopfield(embeddings=embeddings[i*n_embeddings:(i+1)*n_embeddings], input_size=1, num_heads=1, update_steps_max=4, dropout=0.2))

        return networks



def freeze_weights(model):
    for param in model.parameters():
        param.requires_grad = False


# Defining Parameters
def train_autoencoder():
    num_epochs = 30
    model = Autoencoder(base_channel_size=16, latent_dim=128)
    loss_f = nn.MSELoss()
    loss_f.__name__ = "MSE Loss"
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.2,
                                                           patience=10,
                                                           min_lr=5e-5)

    train_epoch = TrainEpoch(model, loss_f, [], optimizer, is_autoencoder=True)
    valid_epoch = ValidEpoch(model, loss_f, [], is_autoencoder=True)

    for epoch in range(num_epochs):
        train_logs = train_epoch.run(train_loader)
        val_logs = valid_epoch.run(valid_loader)
        scheduler.step(val_logs[loss_f.__name__])

    torch.save(model, "autoencoder.pt")

    return model


def get_classification_metrics(average='weighted'):
    metrics = [Accuracy(activation='argmax2d'),
               Precision(activation='argmax2d', average=average),
               Recall(activation='argmax2d', average=average)]
    return metrics


# Defining Parameters
def train_hopfield(autoencoder, embeddings):
    num_epochs = 30
    model = HopfieldModel(autoencoder.encoder, embeddings).cpu()
    loss_f = nn.NLLLoss()
    loss_f.__name__ = "NLL Loss"
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.2,
                                                           patience=10,
                                                           min_lr=5e-5)

    train_epoch = TrainEpoch(model, loss_f, get_classification_metrics("micro"), optimizer, is_autoencoder=False)
    valid_epoch = ValidEpoch(model, loss_f, get_classification_metrics("micro"))

    for epoch in range(num_epochs):
        train_logs = train_epoch.run(train_loader)
        val_logs = valid_epoch.run(valid_loader)
        scheduler.step(val_logs[loss_f.__name__])

    torch.save(model, "hopfield.pt")

    return model


def plot_samples(model, is_autoencoder=False):
    model.eval()
    data_iter = iter(test_loader)
    for _ in range(10):
        x, y = next(data_iter)
        x, y = x[0:1], y[0:1]
        with torch.no_grad():
            if is_autoencoder:
                prediction = model.forward(x)
                plot_two_imgs_t(x.squeeze(), prediction.squeeze())
            else:
                prediction, z = model.forward(x, return_rec_img=True)
                x2 = autoencoder.decoder(z)
                labels = {"y_pr": int(prediction.argmax()), "y_gt": int(y)}
                plot_two_imgs_t(x.squeeze(), x2.squeeze(), labels=labels)


def validate_corrupt(model):
    logs_per_class = {}

    for path in Path("data/mnist_c").iterdir():
        if path.is_dir():
            print(str(path.name))
            loss_f = nn.NLLLoss()
            loss_f.__name__ = "NLL Loss"
            valid_epoch = ValidEpoch(model, loss_f, get_classification_metrics())
            logs_per_class[path.name] = valid_epoch.run(get_corrput_loader(path.name))

    df_data = {"class": [], "NLL Loss": [], "accuracy": [], "precision": [], "recall": []}
    for class_, metrics in logs_per_class.items():
        df_data["class"].append(class_)
        for k, v in metrics.items():
            df_data[k].append(v)

    pd.DataFrame.from_dict(df_data).to_csv("results_corrupted.csv", index=False)


def create_embeddings(encoder, dataloader=train_loader, name="embeddings_train"):
    encoder.eval()
    embeddings = {}
    with tqdm(dataloader, desc="Creating Embeddings...", file=sys.stdout) as iterator:
        for x, y in iterator:
            z = encoder(x)
            for b in range(z.shape[0]):
                y_b, z_b = int(y[b]), z[b]
                if y_b in embeddings:
                    embeddings[y_b].append(z_b)
                else:
                    embeddings[y_b] = [z_b]

    with open(f'{name}.pickle', 'wb') as handle:
        pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return embeddings


def append_class_to_embeddings(embeddings):
    result = []
    for c, embed_list in embeddings.items():
        for embed in embed_list:
            result.append(torch.cat((embed, torch.tensor([c]))))

    random.shuffle(result)
    return torch.stack(result)



if __name__ == '__main__':
    if Path("autoencoder.pt").exists():
        autoencoder = torch.load("autoencoder.pt")
    else:
        autoencoder = train_autoencoder()
        plot_samples(autoencoder)
    freeze_weights(autoencoder)

    if Path("embeddings_train.pickle").exists():
        with open('embeddings_train.pickle', 'rb') as handle:
            embeddings = pickle.load(handle)
    else:
        embeddings = create_embeddings(autoencoder.encoder)
    embeddings = append_class_to_embeddings(embeddings)

    if False and Path("hopfield.pt").exists():
        model = torch.load("hopfield.pt")
    else:
        model = train_hopfield(autoencoder, embeddings)

    plot_samples(model)
    validate_corrupt(model)
