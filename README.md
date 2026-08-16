# DSP-Assisted-Residual-RF-Denoising-

## Overview
Implementation of a DSP pipeline on generated RF Signal Data and removal of Residual AWGN and Rayleigh Fading using a Residual Network


### Pipeline

**Data generation (Simulated):**
- Random Sequential Bits (0/1)
- BPSK and QPSK modulation
- Upsampling (8 sps)
- RRC Pulse shaping

**Channel Impairments (Simulated):**
- Rayleigh Fading
- AWGN Noise

**DSP:**
- Matched Filter
- Low pass Filter
- DC Offset Removal
- Amplitude Normalization



### Repository Structure


```text
DSP-Assisted-Residual-RF-Denoising-/
│
├── data/
│   ├── bpsk/
│   └── qpsk/
├── code/
│   ├── main.py
│   ├── clean_generator.py
│   ├── noise.py
│   ├── dsp.py
│   └── experimental_notebooks/
│
├── requirements.txt
├── README.md
└── .gitignore
```

### Dataset Structure

Both the 'bpsk/' and 'qpsk/' folders contain 'train', 'test' and 'validation' splits.

Each of these folders contains 3 '.npy' files -> 

- {split}_clean.npy - Clean RF signals
- {split}_noisy.npy - Noisy RF signals
- {split}_snr.npy - Signal to Noise ratios for each signal

All the data is stored in '.npy' files of the 'complex64' type, respresenting the I/Q channels.