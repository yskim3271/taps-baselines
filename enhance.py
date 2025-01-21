

import os
import torch
import torchaudio

from matplotlib import pyplot as plt
from utils import LogProgress, mel_spectrogram

def save_wavs(wavs_dict, filepath, sr=16_000):
    for i, (key, wav) in enumerate(wavs_dict.items()):
        torchaudio.save(filepath + f"_{key}.wav", wav, sr)
        
def save_mels(wavs_dict, filepath):
    num_mels = len(wavs_dict)
    figure, axes = plt.subplots(num_mels, 1, figsize=(10, 10))
    figure.set_tight_layout(True)
    figure.suptitle(filepath)
    
    for i, (key, wav) in enumerate(wavs_dict.items()):
        mel = mel_spectrogram(wav, device='cpu', sampling_rate=16_000)
        axes[i].imshow(mel.squeeze().numpy(), aspect='auto', origin='lower')
        axes[i].set_title(key)
    
    figure.savefig(filepath)
    plt.close(figure)

def write(wav, filename, sr=16_000):
    # Normalize audio if it prevents clipping
    wav = wav / max(wav.abs().max().item(), 1)
    torchaudio.save(filename, wav.cpu(), sr)


def enhance(args, model, data_loader, epoch, logger, local_out_dir= None):
    model.eval()
    if local_out_dir:
        out_dir = local_out_dir
    else:
        out_dir = args.out_dir
        
    outdir_mels= os.path.join(out_dir, f"mels_epoch{epoch+1}")
    outdir_wavs= os.path.join(out_dir, f"wavs_epoch{epoch+1}")
    os.makedirs(outdir_mels, exist_ok=True)
    os.makedirs(outdir_wavs, exist_ok=True)

    with torch.no_grad():
        iterator = LogProgress(logger, data_loader, name="Generate enhanced files")
        for data in iterator:
            # Get batch data (batch, channel, time)
            tm, am, tapsId = data
                        
            tapsId = tapsId[0]
            tm = tm.to(args.device)
            am = am.to(args.device)

            am_hat = model(tm)
                        
            tm = tm.squeeze(1).cpu()
            am = am.squeeze(1).cpu()
            am_hat = am_hat.squeeze(1).cpu()
                        
            wavs_dict = {
                "TM": tm,
                "AM": am,
                "AM_hat": am_hat
            }
            
            save_wavs(wavs_dict, os.path.join(outdir_wavs, tapsId))
            save_mels(wavs_dict, os.path.join(outdir_mels, tapsId))
            
            
            
        