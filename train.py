import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

from torchvision import models

from urbansounddataset import UrbanSoundDataset
from cnn import CNNNetwork


BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 1e-5

ANNOTATIONS_FILE = "C:/Users/pasir/Desktop/data/Book1.csv"
AUDIO_DIR = "C:/Users/pasir/Desktop/data/single-channel/enrollment"
SAMPLE_RATE = 16000
NUM_SAMPLES = 16000*3


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)
        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    # instantiating our dataset object and create data loader
    # mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    #     sample_rate=SAMPLE_RATE,
    #     n_fft=512,
    #     hop_length=256,
    #     n_mels=32
    # )
    mel_spectrogram = torchaudio.transforms.Spectrogram(n_fft=512)
    transform = torchaudio.transforms.MFCC(sample_rate=SAMPLE_RATE,n_mfcc=40,
    melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 50, "center": False},)

    usd = UrbanSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            transform,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)
    
    train_dataloader = create_data_loader(usd, BATCH_SIZE)

    # construct model and assign it to device
    # cnn = CNNNetwork().to(device)
    
    
    # state_dict = torch.load("feedforwardnet.pth")
    # cnn.load_state_dict(state_dict)
    print(cnn)
    # initialise loss function + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

    # train model

    train(cnn, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    torch.save(cnn.state_dict(), "feedforwardnet.pth")
    print("Trained feed forward net saved at feedforwardnet.pth")
