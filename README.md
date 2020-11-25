# MultiTune
Adaptive Integration of Multiple Fine-tuning Models in Transfer Learning for Image Classification

**Description of this project:**

This project is an individual research project submitted to The Australian National University (ANU). In this project, a novel technique that can be used in Transfer Learning (TL) is proposed, which enables the adaptive integration of multiple fine-tuning models with different fine-tuning settings. It is denoted as MultiTune. The approach of MultiTune is used in TL for image classification in this project. The datasets used are two datasets from Visual Decathlon Dataset (Rebuffi, Bilen, and Vedaldi, 2017), which are the Aircraft and the Cifar100 datasets. These two datasets are put in the **decathlon-1.0-data folder**. The whole Visual Decathlon Dataset is available at https://www.robots.ox.ac.uk/~vgg/decathlon/#download.

The core blocks of code and the architecture of CNN are based on the paper of SpotTune written by Guo et al. (Guo et al., 2018) Their research on GitHub can be found at https://github.com/gyhui14/spottune. 

**Publication:**
The research paper of this project is accepted by ICONIP2020, and is published on the Springer CCIS 1332 proceedings. The title of the paper is **"MultiTune: Adaptive Integration of Multiple Fine-Tuning Models for Image Classification"** Please use following citation if you are going to use the ideas of this project.

Wang Y., Plested J., Gedeon T. (2020) MultiTune: Adaptive Integration of Multiple Fine-Tuning Models for Image Classification. In: Yang H., Pasupa K., Leung A.CS., Kwok J.T., Chan J.H., King I. (eds) Neural Information Processing. ICONIP 2020. Communications in Computer and Information Science, vol 1332. Springer, Cham. https://doi.org/10.1007/978-3-030-63820-7_56

**Environments Used:**

PyTorch: 1.2.0

Python: 3.6.10

torchvision: 0.4.0

CUDA: 10.1

GPU: Nvidia GTX 1060 (6GB version)

**Instruction for running the code:**

To run the code, run the **main.py** with your IDE. To see the results of MultiTune proposed in this project, just run the code **AS IS**. To compare the SpotTune's result, please turn ***use_multitune*** in **main.py** to **False**. To change the dataset to Cifar100, just download the dataset and put into the correct location, then change ***use_air*** in **main.py** to **False**. To test MultiTune on smaller dataset, turn ***run_small*** in **main.py** to **True** and change ***number_per_class*** (the number of images per class) in line 352 in **main.py**. To get average results for multiple iterations, turn ***run_iteration*** in **main.py** to **True**, and change ***number_of_iteration*** in line 366 to the number of iterations you want to run. 

**References:**

Guo, Yunhui et al. (2018). “SpotTune: Transfer Learning through Adaptive Fine-tuning”. In:CoRRabs/1811.08737. arXiv:1811.08737. URL:http://arxiv.org/abs/1811.08737.

Rebuffi, Sylvestre-Alvise, Hakan Bilen, and Andrea Vedaldi (2017). “Learning multiple visual domains with residual adapters”. In:CoRRabs/1705.08045. arXiv:1705.08045. URL:http://arxiv.org/abs/1705.08045.
