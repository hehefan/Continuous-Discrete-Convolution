# [Continuous-Discrete Convolution](https://openreview.net/forum?id=P5Z-Zl9XJ7)

The structure of proteins involves 3D geometry of amino acid coordinates and 1D sequence of  peptide chains. The 3D structure exhibits irregularity because amino acids are distributed unevenly in Euclidean space and their coordinates are continuous variables. In contrast, the 1D structure is regular because amino acids are arranged uniformly in the chains and their sequential positions (orders) are discrete variables. Moreover, geometric coordinates and sequential orders are in two types of spaces and their units of length are incompatible. These inconsistencies make it challenging to capture the 3D and 1D structures while avoiding the impact of sequence and geometry modeling on each other. This paper proposes a Continuous-Discrete Convolution (CDConv) that uses irregular and regular approaches to model the geometry and sequence structures, respectively. Specifically, CDConv employs independent learnable weights for different regular sequential displacements but directly encodes geometric displacements due to their irregularity. In this way, CDConv significantly improves protein modeling by reducing the impact of geometric irregularity on sequence modeling. Extensive experiments on a range of tasks, including protein fold classification, enzyme reaction  classification, gene ontology term prediction and enzyme commission number prediction, demonstrate the effectiveness of the proposed CDConv. 

CDConv is improved by integrating a few other techniques or tricks. The accuracy can be slightly higher than those numbers reported in the published ICLR23 paper on some tasks. With this work, I would like to suggest that the community unify the modeling for macro 3D (point cloud via LiDAR) and micro 3D (protein or molecule), or employing point cloud techniques to imporve protein or molecule understanding. 

## Installation

The code is tested with Ubuntu 20.04.5 LTS, CUDA v11.7, cuDNN v8.5, PyTorch 1.13.1, PyTorch Geometric (PyG), PyTorch Scatter and PyTorch Sparse. 

Install PyTorch 1.13.1:
```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Install PyG:
```
conda install pyg -c pyg
```

Install PyTorch Scatter and PyTorch Sparse:
```
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
```

## Datasets

We provide the pre-processed datasets for training and evaluating protein representation learning:
1. [Protein Fold](https://drive.google.com/file/d/1vEdezR5L44swsw09WFnaA5zFuA1ZEXHI/view?usp=sharing) &emsp; 2. [Enzyme Reaction](https://drive.google.com/file/d/1eL225Y_6TNYQYlVQNdNOsyK9-bSlDno4/view?usp=sharing) &emsp; 3. [Gene Ontology Term](https://drive.google.com/file/d/1H9zv9vjVXFjR0qjKFTBR3nYSQs3ek0hz/view?usp=sharing) &emsp; 4. [Enzyme Commission Number](https://drive.google.com/file/d/1VEIyBSJbRf9x6k_w4Tqy5SC0G6NWWSWl/view?usp=sharing)

### License
The code is released under MIT License.

### Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{fan2023cdconv,
  title={Continuous-Discrete Convolution for Geometry-Sequence Modeling in Proteins},
  author={Hehe Fan and Zhangyang Wang and Yi Yang and Mohan Kankanhalli},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2023}
}
```

### Related Repos
1. [IEConv](https://github.com/phermosilla/IEConv_proteins) &emsp; 2. [GearNet](https://github.com/DeepGraphLearning/GearNet) &emsp; 3. [PointConv](https://github.com/DylanWusee/pointconv_pytorch) &emsp; 3. [PSTNet](https://github.com/hehefan/Point-Spatio-Temporal-Convolution)
