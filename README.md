# CLE-ViT: Contrastive Learning Encoded Transformer for Ultra-Fine-Grained Visual Categorization

Official PyTorch implementation of [CLE-ViT: Contrastive Learning Encoded Transformer for Ultra-Fine-Grained
Visual Categorization](https) (IJCAI 2023). 

If you use the code in this repo for your work, please cite the following bib entries:

<!--     @article{wang2021feature,
      title={Feature Fusion Vision Transformer for Fine-Grained Visual Categorization},
      author={Wang, Jun and Yu, Xiaohan and Gao, Yongsheng},
      journal={British Machine Vision Conference},
      year={2021}
    } -->


## Abstract
<div style="text-align:justify"> Ultra-fine-grained visual classification (ultra-FGVC) targets at classifying sub-grained categories of fine-grained objects. This inevitably requires discriminative representation learning within a limited training set. Exploring intrinsic features from the object itself, e.g. , predicting the rotation of a given image, has demonstrated great progress towards learning discriminative representation. Yet none of these works consider explicit supervision for learning mutual information at instance level.  To this end, this paper introduces CLE-ViT, a novel contrastive learning encoded transformer, to address the fundamental problem in ultra-FGVC.  The core design is a self-supervised module that performs self-shuffling and masking and then distinguishes these altered images from other images.  This drives the model to learn an optimized feature space
that has a large inter-class distance while remaining tolerant to intra-class variations.  By incorporating this self-supervised module, the network acquires more knowledge from the intrinsic structure of the input data, which improves the generalization ability without requiring extra manual annotations. CLE-ViT demonstrates strong performance on 7 publicly available datasets, demonstrating its effectiveness in the ultra-FGVC task. </div>


<img src='figures/method.jpeg' width='1280' height='350'>


## Create Environment
Please use the command below to create the environment for CLE-ViT.

      $ conda env create -f env.yaml


## Download Google pre-trained ViT models

* [Get models in this link](https://github.com/microsoft/Swin-Transformer): Swin-B, Swin-S...
```bash
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth
```

## Dataset
You can download the datasets from the links below:

+ [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).
+ [Cotton and Soy.Loc](https://drive.google.com/drive/folders/1UkWRepieAvEVEn3Z8n1Zx04bASvvqL7G?usp=sharing).


## Run the experiments.
Using the scripts on scripts directory to train the model, e.g., train on SoybeanGene dataset.

    $ sh scripts/run_gene.sh
    
        
            
## Download Trained Models


[Trained model Google Drive](https://drive.google.com/drive/folders/1g4ex3_P_VOOU5Up_BFdSvrFxVpRQTwc3?usp=share_link)




## Acknowledgment

Our project references the codes in the following repos. Thanks for thier works and sharing.
- [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
- [FFVT](https://github.com/Markin-Wang/FFVT)
