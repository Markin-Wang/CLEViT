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


<img src='architecture.png' width='1280' height='350'>


## Create Environment
Please use the command below to create the environment for CLE-ViT.

      $ conda env create -f py36.yaml


## Download Google pre-trained ViT models

* [Get models in this link](https://console.cloud.google.com/storage/vit_models/): ViT-B_16, ViT-B_32...
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz
```

## Dataset
You can download the datasets from the links below:

+ [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).
+ [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/)
+ [Cotton and Soy.Loc](https://drive.google.com/drive/folders/1UkWRepieAvEVEn3Z8n1Zx04bASvvqL7G?usp=sharing).


## Training scripts for FFVT on Cotton dataset.
Train the model on the Cotton dataset. We run our experiments on 4x2080Ti/4x1080Ti with the batchsize of 8 for each card.

    $ python3 -m torch.distributed.launch --nproc_per_node 4 train.py --name {name} --dataset cotton --model_type ViT-B_16 --pretrained_dir {pretrained_model_dir} --img_size 384 --resize_size 500 --train_batch_size 8 --learning_rate 0.02 --num_steps 2000 --fp16 --eval_every 16 --feature_fusion

## Training scripts for FFVT on Soy.Loc dataset.
Train the model on the SoyLoc dataset. We run our experiments on 4x2080Ti/4x1080Ti with the batchsize of 8 for each card.

    $ python3 -m torch.distributed.launch --nproc_per_node 4 train.py --name {name} --dataset soyloc --model_type ViT-B_16 --pretrained_dir {pretrained_model_dir} --img_size 384 --resize_size 500 --train_batch_size 8 --learning_rate 0.02 --num_steps 4750 --fp16 --eval_every 50 --feature_fusion
    
## Training scripts for FFVT on CUB dataset.
Train the model on the CUB dataset. We run our experiments on 4x2080Ti/4x1080Ti with the batchsize of 8 for each card.

    $ python3 -m torch.distributed.launch --nproc_per_node 4 train.py --name {name} --dataset CUB --model_type ViT-B_16 --pretrained_dir {pretrained_model_dir} --img_size 448 --resize_size 600 --train_batch_size 8 --learning_rate 0.02 --num_steps 10000 --fp16 --eval_every 200 --feature_fusion
    
## Training scripts for FFVT on Dogs dataset.
Train the model on the Dog dataset. We run our experiments on 4x2080Ti/4x1080Ti with the batchsize of 4 for each card.

    $ python3 -m torch.distributed.launch --nproc_per_node 4 train.py --name {name} --dataset CUB --model_type ViT-B_16 --pretrained_dir {pretrained_model_dir} --img_size 448 --resize_size 600 --train_batch_size 4 --learning_rate 3e-3 --num_steps 30000 --fp16 --eval_every 300 --feature_fusion --decay_type linear --num_token 24
    
        
            
## Download  Models


[Trained model Google Drive](https://drive.google.com/drive/folders/1k1vqc0avk_zpCAVuLNZpVX-w-Q3xXf-5?usp=sharing)




## Acknowledgment
Thanks for the advice and guidance given by Dr.Xiaohan Yu and Prof. Yongsheng Gao.

Our project references the codes in the following repos. Thanks for thier works and sharing.
- [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
- [TransFG](https://github.com/TACJu/TransFG)




