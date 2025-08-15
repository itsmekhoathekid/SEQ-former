import torch
import torchaudio
from models import SEQFormerEncoder
from utils import load_json


def debug():

    config_path = "/home/anhkhoa/SEQ-former/configs/local_config.json"

    wav_path = "/mnt/d/voices/voices/VIVOSSPK01_R002.wav"
    waveform, sample_rate = torchaudio.load(wav_path) 
    wav_length = torch.tensor([waveform.shape[0]], dtype=torch.long)

    config = load_json(config_path)
    encoder_config = config['encoder_params']

    seqformer = SEQFormerEncoder(encoder_config)

    x, x_lens, attentions = seqformer(waveform, wav_length)
    print("Input shape:", x.shape)
    print("Input lengths:", x_lens) 
    print("Attention shape:", attentions[0].shape)


if __name__ == "__main__":
    debug()
    print("Debugging completed successfully.")



