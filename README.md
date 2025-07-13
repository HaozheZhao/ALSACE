# NAACL 2024: Mitigating Language-Level Performance Disparity in mPLMs via Teacher Language Selection and Cross-Lingual Self-Distillation

Welcome to the official implementation of the paper "Mitigating Language-Level Performance Disparity in mPLMs via Teacher Language Selection and Cross-Lingual Self-Distillation" presented at NAACL 2024.

## Overview 
This repository demonstrates our proposed approach for addressing the language-level performance disparity issue in multilingual Pretrained Language Models (mPLMs). We introduce the ALSACE that comprises of Teacher Language Selection and Cross-lingual Self-Distillation to mitigate this issue.

# Features
With this code, you can:
- Train the mPLM with the ALSACE method
- Evaluate the language-level performance disparity of the mPLM

# Setup
We conduct our experiment with Anaconda3. If you have installed Anaconda3, then create the environment for ALSACE:
```shell
conda create -n alsace python=3.8.5
conda activate alsace
```

After we setup basic conda environment, install required packages via:
```shell
pip install -r requirements.txt
```

# Training

```shell
bash run_script/finetuning.sh

bash run_script/distillation.sh
```

## Citation
If you use our work, please cite our paper.

```
@misc{zhao2024mitigatinglanguagelevelperformancedisparity,
      title={Mitigating Language-Level Performance Disparity in mPLMs via Teacher Language Selection and Cross-lingual Self-Distillation}, 
      author={Haozhe Zhao and Zefan Cai and Shuzheng Si and Liang Chen and Yufeng He and Kaikai An and Baobao Chang},
      year={2024},
      eprint={2404.08491},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2404.08491}, 
}
```


## License
[Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/)
