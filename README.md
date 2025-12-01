# LAD: LLM-Adapter Enhanced Conditional Diffusion Model

This repository contains the implementation of **LAD** (**L**LM-**A**dapter enhanced conditional **D**iffusion model) for accurate network traffic prediction.

## ⚠️ Important Notice

> **Due to commercial confidentiality and pending patent filings associated with this industry collaboration, only a partial implementation of our model is currently available. The full source code and data will be made publicly available upon patent filing and paper acceptance.**

## 📦 Data & Pretrained Model (Qwen2-0.5B)

Due to the large file size, the dataset and the pretrained LLM weights (**Qwen2-0.5B**) are hosted externally.

- **Download Link**: https://pan.quark.cn/s/3c09c0c4f1ef
- **Password**: `d1xU` 

Please download the files and place them in the appropriate directory before running the model.

## ⚙️ Environment Setup

We recommend using **Anaconda** to manage the environment. Please run the following commands to create the environment and install dependencies:

```bash  
# 1. Create a new conda environment  
conda create -n lad python=3.12 -y  

# 2. Activate the environment  
conda activate lad  

# 3. Install required packages  
pip install -r requirements.txt
```

## 🔧 Configuration

Before running the model, please locate the following variables in the yaml file and update them to match your local paths:

1.  **`data_dir`**: Path to your dataset folder.
2.  **`qwen_model_name`**: Path to the pretrained Qwen2-0.5B weights.

## 🚀 Usage

Please run the following command in the terminal to start the model:

```bash
# 1.For Abilene Dataset
python run.py --config configs/config.yaml
```

## 📂 Project Structure
```text
LAD/
├── configs/                    
│   └── config.yaml
├── model/             
│   ├── __init__.py 
│   └── adapter_enhanced_1117.py
│   └── condition_encoder_enhanced_1117.py
│   └── diffusion_unet_enhanced_1117.py
│   └── qwen_adapter_model_image_enhanced_1117.py
│   └── vision_encoder_enhanced_1117.py
├── scripts/                       
│   ├── __init__.py    
│   └── traffic_image_dataset.py  
├── utils/                       
│   ├── __init__.py    
│   └── metrics.py
│   └── training_1117.py     
├── requirements.txt  
└── README.md  
