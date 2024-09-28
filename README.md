# Auto^6ML Library

## Introduction
Auto^6ML is a open-source library for machine learning automation. It is based entirely on [jittor](https://github.com/Jittor/jittor), offering high performance and faster speeds. The package supports  algorithms based on SLeM(Simulating Learning Methodology ) and some popular meta-learning algorithms.

Our library is divided by methods, which include:
- Data Automation methods 
- Network Automation methods 
- Loss Automation methods  
- Algorithm Automation methods 

## Supported Methods
The currently supported algorithms include:

#### popular meta-learning algorithms
##### Data Automation methods[[Code]](https://github.com/xjtushujun/Auto-6ML/tree/main/MLA/Data%20Automation)
- **L2W**- Learning to Reweight Examples for Robust Deep Learning [[ICML 2018]](https://arxiv.org/pdf/1803.09050) [[Code]](https://github.com/xjtushujun/Auto-6ML/tree/main/MLA/Data%20Automation/L2W)

#### Algorithm based on SLeM

##### Data Automation methods[[Code]](https://github.com/xjtushujun/Auto-6ML/tree/main/SLeM/Data%20Automation)
- **MWNet**- Learning an Explicit Mapping For Sample Weighting [[NeurIPS 2019]](https://arxiv.org/pdf/1902.07379.pdf) [[Code]](https://github.com/xjtushujun/Auto-6ML/tree/main/SLeM/Data%20Automation/MW-Net)
- **PMWNet**- A Probabilistic Formulation for Meta-Weight-Net [[TNNLS 2021]](https://ieeexplore.ieee.org/abstract/document/9525050) [[Code]](https://github.com/xjtushujun/Auto-6ML/tree/main/SLeM/Data%20Automation/PMW-Net)
- **CMWNet**- CMW-Net: Learning a Class-Aware Sample Weighting Mapping for Robust Deep Learning [[TPAMI 2023]](https://arxiv.org/pdf/2202.05613.pdf) [[Code]](https://github.com/xjtushujun/Auto-6ML/tree/main/SLeM/Data%20Automation/CMW-Net)
- **DAC-MR-WNet**- sample weighting for robust deep learning [[submitted]](https://arxiv.org/abs/2305.07892) [[Code]](https://github.com/xjtushujun/Auto-6ML/tree/main/SLeM/Data%20Automation/DAC-MR-WNet)
- **OS3M**- Learning a Trustworthy Sample Selection Strategy
for Open-Set Weakly-Supervised Learning [[submitted]] [[Code]](https://github.com/xjtushujun/Auto-6ML/tree/main/SLeM/Data%20Automation/OS3M)
- **MSLC**- label corrector for noisy labels learning [[AAAI 2021]](https://ojs.aaai.org/index.php/AAAI/article/view/17244) [[Code]](https://github.com/xjtushujun/Auto-6ML/tree/main/SLeM/Data%20Automation/MSLC)
- **CMWNet-SL**- an adaptive robust algorithm for sample selection and label correction [[NSR 2023]](https://academic.oup.com/nsr/article/10/6/nwad084/7086133f) [[Code]](https://github.com/xjtushujun/Auto-6ML/tree/main/SLeM/Data%20Automation/CMW-Net-SL)

##### Network Automation[[Code]](https://github.com/xjtushujun/Auto-6ML/tree/main/SLeM/Data%20Automation)
- **DPIR**- Plug-and-Play Image Restoration With Deep Denoiser Prior [[TPAMI 2022]](https://ieeexplore.ieee.org/abstract/document/9454311) [[Code]](https://github.com/xjtushujun/Auto-6ML/tree/main/SLeM/Network%20Automation/DPIR)
- **L2AC**- Imbalanced Semi-supervised Learning with Bias Adaptive Classifier [[ICLR 2023]](https://arxiv.org/pdf/2207.13856) [[Code]](https://github.com/xjtushujun/Auto-6ML/tree/main/SLeM/Network%20Automation/L2AC)
- **MTA**- Meta Transition Adaptation for Robust Deep Learning with Noisy Labels [[submitted]](https://arxiv.org/pdf/2006.05697.pdf) [[Code]](https://github.com/xjtushujun/Auto-6ML/tree/main/SLeM/Network%20Automation/MTA)

##### Loss Automation[[Code]](https://github.com/xjtushujun/Auto-6ML/tree/main/SLeM/Loss%20Automation)
- **HWNet4ACHN**- Learning an Explicit Weighting Scheme for Adapting Complex HSI Noise [[CVPR 2021]](https://openaccess.thecvf.com/content/CVPR2021/html/Rui_Learning_an_Explicit_Weighting_Scheme_for_Adapting_Complex_HSI_Noise_CVPR_2021_paper.html) [[Code]](https://github.com/xjtushujun/Auto-6ML/tree/main/SLeM/Loss%20Automation/HWNet4ACHN)
- **HWNet4HID**- A Hyper-weight Network for Hyperspectral Image Denoising [[submitted]](https://arxiv.org/pdf/2301.06081.pdf) [[Code]](https://github.com/xjtushujun/Auto-6ML/tree/main/SLeM/Loss%20Automation/HWNet4HID)
- **NARL**- Improve noise tolerance of robust loss via noise-awareness [[TNNLS 2024]](https://arxiv.org/pdf/2301.07306) [[Code]](https://github.com/xjtushujun/Auto-6ML/tree/main/SLeM/Loss%20Automation/NARL-Adjuster)

##### Algorithm Automation[[Code]](https://github.com/xjtushujun/Auto-6ML/tree/main/SLeM/Algorithm%20Automation)
- **MLR**- Improve noise tolerance of robust loss via noise-awareness [[TPAMI 2023]](https://arxiv.org/pdf/2007.14546.pdf) [[Code]](https://github.com/xjtushujun/Auto-6ML/tree/main/SLeM/Algorithm%20Automation/MLR)
- **RG**- Understanding the Generalization of Bilevel Programming in Hyperparameter Optimization: A Tale of Bias-Variance Decomposition [[submitted]] [[Code]](https://github.com/xjtushujun/Auto-6ML/tree/main/SLeM/Algorithm%20Automation/RG)


## Related
### Survey
[1] Jun Shu, Zongben Xu, Deyu Meng. [Small sample learning in big data era.](https://arxiv.org/pdf/1808.04572.pdf) 2018. 

### Framework and Theory
[1] Jun Shu, Deyu Meng, Zongben Xu. [Learning an explicit hyperparameter prediction policy conditioned on tasks.](https://www.jmlr.org/papers/volume24/21-0742/21-0742.pdf) JMLR, 2023.

