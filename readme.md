# Federated Domain Generalization with Generalization Adjustment - CVPR 2023

## Requirements

- Python 3.9.7
- numpy 1.20.3
- torch 1.11.0
- torchvision 0.12.0

## Dataset

Firstly create directory for log files and change the dataset path (`pacs_path`, `officehome_path` and `terrainc_path`) and log path (`log_count_path`) in configs/default.py.
Please download the datasets from the official links:

- PACS [http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017]
- OfficeHome [https://hemanthdv.github.io/officehome-dataset]
- TerraInc [https://beerys.github.io/CaltechCameraTraps]

## Training from scratch

We release the code for PACS dataset and the other two benchmarks can be applied by only changing the dataloader_obj in data/{officehome, terrainc}_dataset.py. All the five FedDG methods are released (FedAvg, FedProx, SCAFFOLD, AM, RSC).

Then running the code:
`
python algorithms/fedavg/train_pacs_GA.py --test_domain p --lr 0.001 --batch_size 16 --comm 40 --model resnet18 --note debug
`

## Acknowledgement

Part of our code is borrowed from the following repositories.

- FACT [https://github.com/MediaBrain-SJTU/FACT]
- DomainBed [https://github.com/facebookresearch/DomainBed]
- FedNova [https://github.com/JYWa/FedNova]
- SCAFFOLD-PyTorch [https://github.com/KarhouTam/SCAFFOLD-PyTorch]
We thank to the authors for releasing their codes. Please also consider citing their works.
