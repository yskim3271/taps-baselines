# Copyright (c) POSTECH, and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: yunsik
import os
import torch
import nlptutti as sarmetric
import numpy as np

from pesq import pesq
from pystoi import stoi
from metric_helper import wss, llr, SSNR, trim_mos
from utils import bold, LogProgress
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def get_stts(args, logger, enhanced):

    cer, wer = 0, 0
    model_id = "ghost613/whisper-large-v3-turbo-korean"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(args.device)
    
    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.float32,
        device=args.device,)
    
    iterator = LogProgress(logger, enhanced, name="STT Evaluation")
    for wav, text in iterator:
        with torch.no_grad():
            transcription = pipe(wav, generate_kwargs={"num_beams": 1, "max_length": 100})['text']

        cer += sarmetric.get_cer(text, transcription, rm_punctuation=True)['cer']
        wer += sarmetric.get_wer(text, transcription, rm_punctuation=True)['wer']
    
    cer /= len(enhanced)
    wer /= len(enhanced)
    
    return cer, wer

## Code modified from https://github.com/wooseok-shin/MetricGAN-plus-pytorch/tree/main
def compute_metrics(target_wav, pred_wav, fs=16000):
    
    Stoi = stoi(target_wav, pred_wav, fs, extended=False)
    Pesq = pesq(ref=target_wav, deg=pred_wav, fs=fs)
        
    alpha = 0.95
    ## Compute WSS measure
    wss_dist_vec = wss(target_wav, pred_wav, 16000)
    wss_dist_vec = sorted(wss_dist_vec, reverse=False)
    wss_dist     = np.mean(wss_dist_vec[:int(round(len(wss_dist_vec) * alpha))])
    
    ## Compute LLR measure
    LLR_dist = llr(target_wav, pred_wav, 16000)
    LLR_dist = sorted(LLR_dist, reverse=False)
    LLRs     = LLR_dist
    LLR_len  = round(len(LLR_dist) * alpha)
    llr_mean = np.mean(LLRs[:LLR_len])
    
    ## Compute the SSNR
    snr_mean, segsnr_mean = SSNR(target_wav, pred_wav, 16000)
    segSNR = np.mean(segsnr_mean)
    
    ## Csig
    Csig = 3.093 - 1.029 * llr_mean + 0.603 * Pesq - 0.009 * wss_dist
    Csig = float(trim_mos(Csig))
    
    ## Cbak
    Cbak = 1.634 + 0.478 * Pesq - 0.007 * wss_dist + 0.063 * segSNR
    Cbak = trim_mos(Cbak)

    ## Covl
    Covl = 1.594 + 0.805 * Pesq - 0.512 * llr_mean - 0.007 * wss_dist
    Covl = trim_mos(Covl)
    
    return Pesq, Stoi, Csig, Cbak, Covl


def evaluate(args, model, data_loader, logger):
        
    metric = {}
    
    iterator = LogProgress(logger, data_loader, name=f"Evaluation")
    enhanced = []
    results  = []
    with torch.no_grad():
        for data in iterator:
            tm, am, id, text = data
            tm = tm.to(args.device)
            am = am.to(args.device)
            
            am_hat = model(tm)
                        
            am_hat = am_hat.squeeze().cpu().numpy()
            am = am.squeeze().cpu().numpy()
            
            enhanced.append((am_hat, text[0]))
            results.append(compute_metrics(am, am_hat))
    
    results = np.array(results)
    pesq, stoi, csig, cbak, covl = np.mean(results, axis=0)
    metric = {
        "pesq": pesq,
        "stoi": stoi,
        "csig": csig,
        "cbak": cbak,
        "covl": covl
    }
    logger.info(bold(f"Performance: PESQ={pesq:.4f}, STOI={stoi:.4f}, CSIG={csig:.4f}, CBAK={cbak:.4f}, COVL={covl:.4f}"))
    
    if args.eval_stt:
        cer, wer = get_stts(args, logger, enhanced)
        metric['cer'] = cer
        metric['wer'] = wer
        logger.info(bold(f"Performance: CER={cer:.4f}, WER={wer:.4f}"))
   
    return metric



if __name__=="__main__":
    import logging
    import logging.config
    import argparse
    import importlib
    from data import TAPSdataset
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from datasets import load_dataset
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the model configuration file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint directory. default is current directory")
    parser.add_argument("--log_file", type=str, default="evaluate.log", help="Name of the log file. default is evaluate.log")
    parser.add_argument("--eval_stt", default=True, action="store_true", help="Evaluate the model using the STT model.")
    parser.add_argument("--device", type=str, default="cuda", help="Specifies the device (cuda or cpu).")

    args = parser.parse_args()
    
    checkpoint_path = args.checkpoint
    device = args.device
    log_file = args.log_file

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    try:
        conf = OmegaConf.load(args.config)
        conf.device = device
        conf.eval_stt = args.eval_stt
                
        model_name = conf.model_name
        module = importlib.import_module("models."+ model_name)
        model_class = getattr(module, model_name)
        model = model_class(**conf.param).to(device)
        chkpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(chkpt['model'])
        
        testset = load_dataset("yskim3271/Throat_and_Acoustic_Pairing_Speech_Dataset", split="test")
        
        tt_dataset = TAPSdataset(datapair_list=testset, with_id=True, with_text=True)
        tt_loader = DataLoader(
            dataset=tt_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True
            )
            
        logger.info(f"Model: {model_name}")
        logger.info(f"Checkpoint: {checkpoint_path}")
        logger.info(f"Device: {device}")
        
        model.eval()
        evaluate(args=conf, 
                model=model,
                data_loader=tt_loader,
                logger=logger)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise e