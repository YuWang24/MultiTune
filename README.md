# MultiTune
Adaptive Integration of Multiple Fine-tuning Models in Transfer Learning for Image Classification

**Description of this porject:**

This project is an individual research project submitted to The Autralian National University (ANU). In this project, a novel technique that can be used in Transfer Learning (TL) is proposed, which enables the adaptive integration of multiple fine-tuning models with different fine-tuning settings. It is denoted as MultiTune. The approach of MultiTune is used in TL for image classification in this project. The datasets used are two datasets from Visual Decathlon Dataset (Rebuffi, Bilen, and Vedaldi, 2017), which are the Aircraft and the Cifar100 datasets. Due to the size limit, only the Aircraft dataset is uploaded in this GitHub project. The Cifar100 dataset can be donwloaded from https://www.robots.ox.ac.uk/~vgg/decathlon/#download. After donwloading, put the dataset in the **decathlon-1.0-data folder**.

The core blocks of code and the architecture of CNN are based on the paper of SpotTune written by Guo et al. (Guo et al., 2018) Their research on GitHub can be found on https://github.com/gyhui14/spottune. 

**Environments Used:**

PyTorch: 1.2.0

Python: 3.6.10

torchvision: 0.4.0

CUDA: 10.1

GPU: Nvidia GTX 1060 (6GB version)

**Instruction for running the code:**

To run the code, run the **main.py** with your IDE. To see the results of MultiTune proposed in this project, just run the code **AS IS**. To compare the SpotTune's result, please turn ***use_multitune*** in **main.py** to **False**. To change the dataset to Cifar100, just download the dataset and put into the correct location, then change ***use_air*** in **main.py** to **False**. To test MultiTune on smaller dataset, turn ***run_small*** in **main.py** to **True** and change ***number_per_class*** (the number of images per class) in line 352 in **main.py**. To get average results for multiple iterations, turn ***run_iteration*** in **main.py** to **True**, and change ***number_of_iteration*** in line 366 to the number of iterations you want to run. 

**References:**

Guo, Yunhui et al. (2018). “SpotTune: Transfer Learning through AdaptiveFine-tuning”. In:CoRRabs/1811.08737. arXiv:1811.08737. URL:http://arxiv.org/abs/1811.08737.

Rebuffi, Sylvestre-Alvise, Hakan Bilen, and Andrea Vedaldi (2017). “Learn-ing multiple visual domains with residual adapters”. In:CoRRabs/1705.08045. arXiv:1705.08045. URL:http://arxiv.org/abs/1705.08045.
