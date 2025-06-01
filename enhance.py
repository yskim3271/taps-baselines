

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
    if local_out_dir:
        out_dir = local_out_dir
    else:
        out_dir = args.out_dir
        
    suffix = f"_epoch{epoch+1}" if epoch is not None else ""
        
    outdir_mels= os.path.join(out_dir, f"mels" + suffix)
    outdir_wavs= os.path.join(out_dir, f"wavs" + suffix)
    os.makedirs(outdir_mels, exist_ok=True)
    os.makedirs(outdir_wavs, exist_ok=True)

    with torch.no_grad():
        if logger is not None:
            iterator = LogProgress(logger, data_loader, name="Generate enhanced files")
        else:
            iterator = data_loader
        for data in iterator:
            # Get batch data (batch, channel, time)
            tm, am, id, _ = data
                        
            id = id[0]
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
            
            save_wavs(wavs_dict, os.path.join(outdir_wavs, id))
            save_mels(wavs_dict, os.path.join(outdir_mels, id))
            
            

if __name__=="__main__":
    import logging
    import logging.config
    import argparse
    import importlib
    from data import TAPSdataset, StepSampler
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from datasets import load_dataset
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the model configuration file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint directory. default is current directory")
    parser.add_argument("--output_dir", type=str, default="samples", help="Output directory for enhanced samples. default is samples")
    parser.add_argument("--device", type=str, default="cuda", help="Specifies the device (cuda or cpu).")
    
    args = parser.parse_args()
    checkpoint_path = args.checkpoint
    device = args.device
    local_out_dir = args.output_dir

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        conf = OmegaConf.load(args.config)
        conf.device = device
        
        model_args = conf.model
        model_name = model_args.model_name
        module = importlib.import_module("models."+ model_name)
        model_class = getattr(module, model_name)
        model = model_class(**model_args.param).to(device)
        chkpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(chkpt['model'])
        
        testset = load_dataset("yskim3271/Throat_and_Acoustic_Pairing_Speech_Dataset", split="test")
        
        tt_dataset = TAPSdataset(datapair_list=testset, with_id=True, with_text=True)
        tt_loader = DataLoader(
            dataset=tt_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            sampler=StepSampler(len(tt_dataset), step=100)
        )
        
        logger.info(f"Model: {model_name}")
        logger.info(f"Checkpoint: {checkpoint_path}")
        logger.info(f"Device: {device}")
        logger.info(f"Output directory: {local_out_dir}")
        os.makedirs(local_out_dir, exist_ok=True)
        
        model.eval()
        enhance(args=conf,
                model=model,
                data_loader=tt_loader,
                epoch=None,
                logger=logger,
                local_out_dir=local_out_dir
                )
    except Exception as e:
        logger.error(f"Error: {e}")
        raise e
    
    
    