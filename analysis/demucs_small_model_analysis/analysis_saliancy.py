from demucs import pretrained, htdemucs
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import random
from pytorch_lightning.loggers import WandbLogger
import wandb
import auraloss
import collections
from tqdm import tqdm
import pretty_midi
import matplotlib.pyplot as plt
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB
print(torch.cuda.is_available())
from torch.optim import lr_scheduler
from IPython.display import Audio
from torchaudio.transforms import Fade
import musdb
import museval
import gc
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import random
from pytorch_lightning.loggers import WandbLogger
import subprocess
import wandb
import auraloss
import collections
from tqdm import tqdm
import librosa
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB
print(torch.cuda.is_available())
from torch.optim import lr_scheduler
import pretty_midi
from typing import Optional, Tuple
import pandas as pd

seed_value = 3407
torch.manual_seed(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.set_float32_matmul_precision('high')

path = "/import/c4dm-datasets/musdb18hq_v2/"
path_adtof = "/homes/hsk01/drum_sep/data/adtof/adtof/"

train = list(os.listdir(path+'train'))
test = list(os.listdir(path+'test'))

all_scenes = {}
counter = 0
length_s = 4.0
for idx, val in tqdm(enumerate(test)):
    p = path + 'test/' + val + "/"
    p2 = path_adtof + 'test/' + val + "/adtof.mid"
    
    info = torchaudio.info(f"{p}mixture.wav")
    seconds = info.num_frames // 44100
    for i in range(0, seconds - (1 + int(length_s//1)), 4):
        start_point = int(i * 44100)
        if start_point + ((44100 * length_s)) < info.num_frames:
            all_scenes[counter] = {'music_path': p, 'adtof_path': p2, 'track_name':val ,'start_point': start_point, 'length':  (44100 * length_s), 'frames' : info.num_frames}
            counter += 1

def turn_transcription_into_roll(transcription, frames):
    # Determine your sampling frequency (frames per second)
    fs = 44100
    
    piano_roll_length = int(frames)
    
    # Initialize the piano roll array
    piano_roll = np.zeros((64, piano_roll_length))
    
    # Fill in the piano roll array
    for note in transcription.instruments[0].notes:
        # Convert start and end times to frame indices
        start_frame = int(np.floor(note.start * fs))
        end_frame = int(np.ceil(note.end * fs))
        
        # Set the corresponding frames to 1 (or note.velocity for a velocity-sensitive representation)
        piano_roll[note.pitch, start_frame:end_frame] = 1  # Or use note.velocity
        
    roll = np.vstack([piano_roll[35:36, :], piano_roll[38:39, :], piano_roll[42:43, :], piano_roll[47:48, :], piano_roll[49:50, :]])
    return roll

# # making the model
class NewSDRLoss(nn.Module):
    """
    New Signal-to-Distortion Ratio (SDR) loss module based on the MDX challenge definition.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none': no reduction will be applied,
            'mean': the mean of the output will be calculated,
            'sum': the sum of the output will be calculated. Default: 'mean'.
    """
    def __init__(self, reduction='mean'):
        super(NewSDRLoss, self).__init__()
        self.reduction = reduction

    def forward(self, estimates, references):
        """
        Computes the SDR loss between the estimated and reference signals.

        Args:
            estimates (Tensor): Estimated signals, shape (batch, channels, height, width).
            references (Tensor): Reference signals, shape (batch, channels, height, width).

        Returns:
            Tensor: The calculated SDR loss.
        """
        assert references.dim() == 4 and estimates.dim() == 4, "Inputs must be 4D tensors."

        delta = 1e-7  # Avoid numerical errors
        num = torch.sum(torch.square(references), dim=(2, 3))
        den = torch.sum(torch.square(references - estimates), dim=(2, 3))
        num += delta
        den += delta
        scores = 10 * torch.log10(num / den)

        if self.reduction == 'mean':
            return -scores.mean()
        elif self.reduction == 'sum':
            return -scores.sum()
        else:  # 'none'
            return -scores

class DrumDemucs(pl.LightningModule):
    def __init__(self):
        super(DrumDemucs, self).__init__()

        self.loss_fn = auraloss.freq.MultiResolutionSTFTLoss(
                    fft_sizes=[1024, 2048, 4096],
                    hop_sizes=[256, 512, 1024],
                    win_lengths=[1024, 2048, 4096],
                    scale="mel", 
                    n_bins=150,
                    sample_rate=44100,
                    device="cuda"
                )

        self.loss_fn_2 = auraloss.time.SISDRLoss()

        self.loss_fn_3 = torch.nn.L1Loss()

        self.loss_fn_4 = NewSDRLoss()

        self.loss_used = 0

        sources = ['drum',
                   'noise',
                   ]
        
        self.demucs_mixer =  torchaudio.models.HDemucs(
            sources=sources,
            audio_channels=7,
            depth=6,
        )

        self.out_conv = nn.Conv1d(in_channels=7, out_channels=2, kernel_size=1)
        self.out = nn.Conv1d(in_channels=2, out_channels=2, kernel_size=1)      

    def compute_loss(self, outputs, ref_signals):
        # loss = self.loss_fn(outputs, ref_signals) + self.loss_fn_2(outputs, ref_signals) +  self.loss_fn_3(outputs, ref_signals)
        loss = self.loss_fn_4(outputs.unsqueeze(2), ref_signals.unsqueeze(2))
        return loss

    def forward(self, audio, drumroll):
        to_mix = torch.cat([audio, drumroll], axis=1)
        out = self.demucs_mixer(to_mix)
        out_2 = self.out_conv(out[:, 0, :, :])
        out_2 = self.out(out_2)
        # out_2 = torch.tanh(out_2)

        return out_2
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        audio, drum, drumroll = batch
        
        outputs = self.forward(audio, drumroll)
        # print(outputs.size())

        if batch_idx % 64 == 0:
            input_signal = audio[0].cpu().detach().numpy().T
            generated_signal = outputs[0].cpu().detach().numpy().T
            drum_signal = drum[0].cpu().detach().numpy().T 
            wandb.log({'audio_input': [wandb.Audio(input_signal, caption="Input", sample_rate=44100)]})
            wandb.log({'audio_reference': [wandb.Audio(drum_signal, caption="Reference", sample_rate=44100)]})
            wandb.log({'audio_output': [wandb.Audio(generated_signal, caption="Output", sample_rate=44100)]})
             
            for i in range(5):
                wandb.log({f'drum_{i + 1}': [wandb.Audio(drumroll[0].cpu().detach().numpy()[i, :], caption="Output", sample_rate=44100)]})


        loss = self.compute_loss(outputs, drum)         

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    

    def configure_optimizers(self):
        # Define your optimizer and optionally learning rate scheduler here
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return [optimizer]
        
 
def load_audio(path, start_point, filename):
    file = filename
    start_seconds = start_point // 44100
    duration = 4

    segment, _ = librosa.load(f"{path}/{file}", sr=44100, mono=False, offset=start_seconds, duration=duration)
    segment_tensor = torch.tensor(segment)

    return segment_tensor

def load_roll(path, start_point, frames):
    midi = path
    transcription = pretty_midi.PrettyMIDI(midi)
    roll = turn_transcription_into_roll(transcription, frames)
    # print(roll.shape)
    roll = roll[:, start_point: start_point + (44100 * 4)]
    return torch.from_numpy(roll).float()

def separate_sources(
    model,
    mix,
    drumroll,
    segment=4.0,
    overlap=0,
    device='cuda',
):
    """
    Apply model to a given mixture. Use fade, and add segments together in order to add model segment by segment.

    Args:
        segment (int): segment length in seconds
        device (torch.device, str, or None): if provided, device on which to
            execute the computation, otherwise `mix.device` is assumed.
            When `device` is different from `mix.device`, only local computations will
            be on `device`, while the entire tracks will be stored on `mix.device`.
    """
    if device is None:
        device = mix.device
    else:
        device = torch.device(device)

    batch, channels, length = mix.shape
    sample_rate = 44100
    chunk_len = int(sample_rate * segment * (1 + overlap))
    start = 0
    end = chunk_len
    overlap_frames = overlap * sample_rate
    fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear")

    final = torch.zeros(batch, channels, length, device=device)

    while start < length - overlap_frames:
        chunk = mix[:, :, start:end]
        roll = drumroll[:, :, start:end]
        roll = torch.ones_like(roll).to(model.device)
        with torch.no_grad():
            out = model.forward(chunk, roll)
        out = fade(out)
        final[:, :, start:end] += out
        if start == 0:
            fade.fade_in_len = int(overlap_frames)
            start += int(chunk_len - overlap_frames)
        else:
            start += chunk_len
        end += chunk_len
        if end >= length:
            fade.fade_out_len = 0
    return final

class AudioData:
    def __init__(self, audio):
        self.audio = audio


model = DrumDemucs.load_from_checkpoint(f'/homes/hsk01/drum_sep/htdemucs/checkpoint/epoch_280.ckpt')
model.to('cuda')
# model = model.eval()
    
idxs = list(all_scenes)

sdr = {}
sdr_underlying = {}
current_track = ""

saliance_rows = []
for idx in tqdm(idxs):
    sample =  all_scenes[idx]
    audio_path = sample['music_path']
    adtof_path = sample['adtof_path']

    start_point = sample['start_point']

    mixture_tensor = load_audio(audio_path, start_point,'mixture.wav').unsqueeze(0).to(model.device).requires_grad_(True)
    drum_tensor = load_audio(audio_path, start_point,'drums.wav').unsqueeze(0).to(model.device).requires_grad_(True)
    roll_tensor = load_roll(adtof_path, start_point, sample['frames']).unsqueeze(0).to(model.device).requires_grad_(True)
    sep = model(mixture_tensor, roll_tensor)
    target = sep.norm()

    model.zero_grad()
    target.backward()
    saliency_mix  = mixture_tensor.grad.abs().squeeze(0).mean(dim=1)
    saliency_roll = roll_tensor.grad.abs().squeeze(0).mean(dim=1)
    
    # print(saliency_mix.shape, saliency_roll.shape)
    # print(saliency_mix, saliency_roll)

    row = [
        sample['track_name'], sample['start_point'],
        saliency_mix[0].item(), saliency_mix[1].item(),  # l, r from saliency_mix
        saliency_roll[0].item(), saliency_roll[1].item(), saliency_roll[2].item(),  # d1, d2, d3 from saliency_roll
        saliency_roll[3].item(), saliency_roll[4].item(),  # d4, d5 from saliency_roll
    ]
    saliance_rows.append(row)
    # break

df = pd.DataFrame(saliance_rows, columns=['track_name', 'start_point', 'l', 'r', 'd1', 'd2', 'd3', 'd4', 'd5'])
df.to_csv('/homes/hsk01/drum_sep/htdemucs/saliance.csv')
print(df)