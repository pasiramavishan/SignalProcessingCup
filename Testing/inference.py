import torch
import torchaudio
from urbansounddataset import UrbanSoundDataset
from cnn import CNNNetwork
from train import AUDIO_DIR, SAMPLE_RATE, NUM_SAMPLES, ANNOTATIONS_FILE


class_mapping = [
"spk_2",
"spk_3",
"spk_4",
"spk_5",
"spk_6",
"spk_8",
"spk_9",
"spk_10",
"spk_11",
"spk_12",
"spk_13",
"spk_14",
"spk_15",
"spk_16",
"spk_17",
"spk_18",
"spk_19",
"spk_20",
"spk_21",
"spk_22",
"spk_23",
"spk_24",
"spk_25",
"spk_26",
"spk_27",
"spk_28",
"spk_29",
"spk_30",
"spk_31",
"spk_32",
"spk_33",
"spk_34",
"spk_35",
"spk_36",
"spk_37",
"spk_38",
"spk_39",
"spk_40",
"spk_41",
"spk_42",
"spk_43",
"spk_44",
"spk_45",
"spk_46",
"spk_47",
"spk_48",
"spk_49",
"spk_50",
"spk_51",
"spk_52",
"spk_53",
"spk_54",
"spk_55",
"spk_56",
"spk_57",
"spk_58",
"spk_59",
"spk_60",
"spk_61",
"spk_62",
"spk_63",
"spk_64",
"spk_65",
"spk_66",
"spk_67",
"spk_68",
"spk_69",
"spk_70",
"spk_71",
"spk_72",
"spk_73",
"spk_74",
"spk_75",
"spk_76",
"spk_77"
]


def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected


if __name__ == "__main__":
    # load back the model
    cnn = CNNNetwork()
    state_dict = torch.load("feedforwardnet.pth")
    cnn.load_state_dict(state_dict)

    #load urbansound dataset
    # mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    #     sample_rate=SAMPLE_RATE,
    #     n_fft=1024,
    #     hop_length=512,
    #     n_mels=64
    # )
    mel_spectrogram = torchaudio.transforms.Spectrogram(n_fft=512)
    transform = torchaudio.transforms.MFCC(sample_rate=SAMPLE_RATE,n_mfcc=40,
    melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 50, "center": False},)

    usd = UrbanSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            transform,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            "cpu")

    # get a sample from the urban sound dataset for inference
    m = 0
    # input, target = usd[24][0], usd[24][1]  #[batch size, num_channels, fr, time]
    # input.unsqueeze_(0)
    # predicted, expected = predict(cnn, input, target, class_mapping)
    # print(predicted, expected)
    for i in range(76):
        input, target = usd[i][0], usd[i][1]  #[batch size, num_channels, fr, time]
        input.unsqueeze_(0)
        

        # make an inference
        predicted, expected = predict(cnn, input, target, class_mapping)
        if predicted == expected:
            m = m + 1
        print(f"total: {i}, correct: {m} ")
