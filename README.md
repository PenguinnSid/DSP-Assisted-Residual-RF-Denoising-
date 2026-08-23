# DSP-Assisted-Residual-RF-Denoising-

## Overview
Implementation of a DSP pipeline on generated RF Signal Data and removal of Residual AWGN and Rayleigh Fading using a Residual Network


## Pipeline

**Data generation (Simulated):**
- Random Sequential Bits (0/1)
- BPSK and QPSK modulation
- Upsampling (8 sps)
- RRC Pulse shaping

**Channel Impairments (Simulated):**
- Rayleigh Fading
- AWGN Noise

**DSP:**
- DC Offset Removal
- Matched Filter
- Low pass Filter
- Amplitude Normalization



## Repository Structure


```text
DSP-Assisted-Residual-RF-Denoising-/
│
├── data/
│   ├── bpsk/
│   ├── qpsk/
│   └── rrc_filter.npy
│
├── code/
│   ├── main.py
│   ├── clean_generator.py
│   ├── noise.py
│   ├── dsp.py
│   └── experimental_notebooks/
│
├── tests/
│   └── dsp_test.py/
│ 
├── requirements.txt
├── README.md
└── .gitignore
```

## Data Structure

Both the 'bpsk/' and 'qpsk/' folders contain 'train', 'test' and 'validation' splits.

Each of these folders contains 8 '.npy' files -> 

- {split}_clean.npy - Clean RF signals
- {split}_dsp.npy - Signals after applying DSP
- {split}_faded_target.npy - faded signal passed through the LPF and normalized (without AWGN)
- {split}_faded.npy - Signals after Rayleigh fading is applied
- {split}_h.npy - Rayleigh fading coefficients
- {split}_noisy.npy - Faded signal with AWGN
- {split}_snr.npy - Signal to Noise ratios for each signal
- {split}_target.npy - Clean signals passed through Matched filter, Low Pass Filter and normalized

All the data is stored in '.npy' files of the 'complex64' type, respresenting the I/Q channels.

## Running

### Setup

Clone the repository

```bash
git clone https://github.com/PenguinnSid/DSP-Assisted-Residual-RF-Denoising-.git
```

Change the directory

```bash
cd DSP-Assisted-Residual-RF-Denoising-
```

Setup the virtual enviorment

```bash
python -m venv venv
venv\Scripts\activate
```

Install the dependencies

```bash
pip install -r requirements.txt
```

## Data Generation and Tests

To generate the data, save it and apply the simulated noise and DSP

```bash
python main.py 
```

To test the DSP pipeline and check the MSE and SNR

```bash
python main.py 
```
