# deconstructing-bias-skin-lesion

Code to reproduce the results for the paper "(De)Constructing Bias on Skin Lesions Datasets" in ISIC Skin Image Analysis Workshop @ CVPR 2019. [Link to the paper.](https://arxiv.org/pdf/1904.08818.pdf)

# Preparing data and environment
### Configuring the container.
We used nvidia-docker for all experiments. Run the following command to configure a container:

`nvidia-docker run -ti --userns=host --shm-size 8G  -v /home/deconstructing-bias-skin-lesion/:/deconstructing-bias-skin-lesion/ --name deconstructingbias nvidia/cuda:9.1-devel-ubuntu16.04 /bin/bash`

### Inside the container, install dependencies:
  `apt-get install imagemagick git python3 python3-pip`
  
  `pip3 install -r requirements.txt`
  
### Download and extract your data.
We used the Interactive Atlas of Dermoscopy and data from the ISIC Archive for our experiments.
You need to download the [2018 ISIC Challenge](https://challenge2018.isic-archive.com/participate/) training set. Download images and ground truth for tasks 1 and 2.

To create the disturbed, and the attribute sets used in our experiments, please check the `scripts` folder.

## Run experiments!
All the exact splits used are available in folders `atlas-csv` and `isic-csv`. 
To train and evaluate the network, refer to the scripts `run_isic_7030_all.sh` and `run_isic_rgbm_7030.sh`.

## Citation
```
@inproceedings{bissoto19deconstructing,
 author    = {Alceu Bissoto and Michel Fornaciali and Eduardo Valle and Sandra Avila},
 title     = {({D}e){C}onstructing Bias on Skin Lesion Datasets},
 booktitle = {ISIC Skin Image Anaylsis Workshop, 2019 {IEEE} Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
 year      = {2019},
}
```
## Acknowledgments
A. Bissoto and S. Avila are partially funded Google LARA 2018. A. Bissoto is also partially funded by CNPq. E. Valle is partially funded by a CNPq PQ-2 grant (311905/2017-0). This work was funded by grants from CNPq (424958/2016-3), FAPESP (2017/16246-0) and FAEPEX (3125/17). The RECOD Lab receives addition funds from FAPESP, CNPq, and CAPES. We gratefully acknowledge NVIDIA for the donation of GPU hardware.
