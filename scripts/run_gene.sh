export tag='gene_sb7_fullb_np2_lr2e-3_mul5_bs12_448_ep200_linear_ee2'
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py \
--dataset soybean_gene \
--tag $tag \
--lr 2e-3 \
--model full \
--mask \
--swap \
--con \
--margin 1 \
--origin_w 1 \
--swap_w 0.3 \
--con_w 0.5 \
--num_part 2 \
--img_size 448 \
--cfg configs/swin/swin_base_patch4_window7_448.yaml \
--data-path ./datasets \
--batch-size 12 \
--eval-batch-size 36 \
--epochs 200 \
--pretrained ./pretrained_models/swin_base_patch4_window7_224_22k.pth \
>> logs/gene/$tag 2>&1 & \
tail -fn 50 logs/gene/$tag
#--mask \
#--swap \
#--con \
#--margin 1 \
#--origin_w 1 \
#--swap_w 1 \
#--con_w 1 \
#--num_part 2 \

