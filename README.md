# Code for training baseline models on TAPS dataset
We provide the code for training baseline models on the TAPS dataset. The code includes the implementation of the Demucs, TSTNN, and SE-Conformer models. The code is written in PyTorch and uses the Huggingface Datasets library for data loading.

## TAPS Dataset
**The TAPS (Throat and Acoustic Pairing Speech)** dataset is a unique speech dataset designed to support deep learning-based throat microphone speech enhancement. It consists of paired utterances recorded using both throat microphones (capturing vibrations from the neck) and acoustic microphones (recording airborne speech). The dataset aims to improve speech clarity in high-noise environments where conventional microphones struggle.

The dataset is publicly available and can be downloaded from the **Huggingface repository**: [TAPS Dataset](https://huggingface.co/datasets/yskim3271/Throat_and_Acoustic_Pairing_Speech_Dataset) 🤗

For more information on the dataset, please visit our **project page** on [http://taps.postech.ac.kr/](http://taps.postech.ac.kr)

## Baseline Models  

#### **Demucs**  
Demucs is a deep learning-based speech enhancement model that employs a convolutional encoder-decoder architecture with U-Net-style skip connections. It integrates a bidirectional LSTM for sequence modeling, effectively capturing temporal dependencies in speech. The model is particularly effective at reconstructing lost speech features and improving intelligibility.  

📄 **Reference:** [Demucs GitHub Repository](https://github.com/facebookresearch/denoiser)  

#### **TSTNN**  
TSTNN (Two-Stage Transformer-based Neural Network) is a masking-based model designed for time-domain speech enhancement. It utilizes a two-stage transformer module (TSTM) consisting of local and global transformers. These modules refine speech representations by preserving important components while suppressing noise. However, the model struggles to reconstruct missing voiceless sounds, limiting its effectiveness in certain speech enhancement tasks.  

📄 **Reference:** [TSTNN Paper on arXiv](https://arxiv.org/abs/2103.09963) | [TSTNN GitHub Repository](https://github.com/key2miao/TSTNN)  

#### **SE-Conformer**  
SE-Conformer is a speech enhancement model that combines convolutional and transformer-based architectures. It employs a Conformer block for sequence modeling, capturing both local and global dependencies in speech signals. The model demonstrates strong performance in enhancing speech clarity, particularly by effectively reconstructing high-frequency components and voiceless sounds.  

📄 **Reference:** [SE-Conformer Paper (ISCA Archive)](https://www.isca-archive.org/interspeech_2021/kim21h_interspeech.html)  

## Installation
`pip install requirements.txt`

## Running Evaluation on the Test Set
The `evaluate.py` script is used to evaluate the model on the test set. The script will evaluate the model performace in terms of PESQ, STOI, CSIG, CBAK, and COVL. You can also evaluate the model using the STT model to calculate the Character Error Rate (CER) and Word Error Rate (WER). The default model for the STT evaluation is the [whisper-large-v3-turbo-korean](https://huggingface.co/ghost613/whisper-large-v3-turbo-korean) model which is a fine-tuned version of the [whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) model on the Korean language. You can change the model by changing the `enhance.py` script.
```
python evaluate.py --ckpt_dir=<path to the checkpoint> --log_file=<filename of log>
```

For more details regarding possible arguments, please see:
```
usage: evaluate.py [-h] [--ckpt_dir CKPT_DIR] [--ckpt_file CKPT_FILE] [--log_file LOG_FILE] [--device DEVICE] [--eval_stt]

optinal arguments:
  -h, --help                show this help message and exit
  --ckpt_dir CKPT_DIR       Path to the model checkpoint directory.
  --ckpt_file CKPT_FILE     Checkpoint file name. default is "best.th"
  --log_file LOG_FILE       Name of the log file. default is "evalute.log"
  --eval_stt                Evaluate the model using the STT model.
  --device DEVICE           Specifies the device (cuda or cpu). default is "cuda" if available.
```


## Inference on the Test Set
The `enhance.py` script is used to enhance the audio samples in the test set.
```
python enhance.py --ckpt_dir=<path to the checkpoint> --output_dir=<sample directory>
```
For more details regarding possible arguments, please see:

```
usage: enhance.py [-h] [--ckpt_dir CKPT_DIR] [--ckpt_file CKPT_FILE] [--output_dir OUTPUT_DIR] [--device DEVICE]

optinal arguments:
  -h, --help                show this help message and exit
  --ckpt_dir CKPT_DIR       Path to the model checkpoint directory.
  --ckpt_file CKPT_FILE     Checkpoint file name. default is best.th
  --output_dir OUTPUT_DIR   Directory to save the enhanced samples. default is "samples"
  --device DEVICE           Specifies the device (cuda or cpu). default is "cuda" if available.
```


## Hydra
We use Hydra to manage the configuration files. Hydra is a powerful tool for elegantly configuring complex applications. It composes and overrides configurations in a flexible way, making it easy to run multiple experiments with different configurations. See the [Hydra Documentation](https://hydra.cc/docs/intro) for more information.

## Train baseline models
The `train.py` script is used to train the model on the TAPS dataset. The script will save the model checkpoints, logs, and tensorboard logs in the `outputs` directory. You can also override any parameter in the configuration file using the command line. 

The configuration files are stored in the `configs` directory. You can train baseline models by selecting the configuration file for the model you want to train. For example, to train the SE-Conformer model, you can use the `config_seconformer.yaml` configuration file. If you want to change model parameters, you can override model configs in `conf/model` directory.

```
python train.py --config_name="config_$ModelName" $OverrideParams
```
- `--config_name`: (required, str) – Name of the configuration file to use.
- `$OverrideParams`: (optional) – Override any parameter in the configuration file. For example, `data.batch_size=16` will override the batch_size parameter in the configuration file.

## Train using your own dataset
The default dataset used in this code is the TAPS dataset. You can train the model with your own dataset by changing the `data.py` and `train.py` scripts.

## Download Pretrained Models
You can download the pretrained models from the following links:
[Google Drive](https://drive.google.com/drive/folders/133hBcBob8wJ-WaV7qLNj9G3eqdzyd2vX?usp=drive_link)