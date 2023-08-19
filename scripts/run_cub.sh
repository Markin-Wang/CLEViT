export tag='cub_sb_fullb_np2_m1_lr1e-3_mul5_bs12_448_ep40_ee1'
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py \
--dataset CUB \
--tag $tag \
--lr 1e-3 \
--model full \
--mask \
--swap \
--con \
--margin 1 \
--origin_w 1 \
--swap_w 0.3 \
--con_w 0.5 \
--num_part 2 \
--head_mul 5 \
--img_size 448 \
--cfg configs/swin/swin_base_patch4_window7_448.yaml \
--data-path ./datasets \
--batch-size 12 \
--eval-batch-size 36 \
--epochs 40 \
--pretrained ./pretrained_models/swin_base_patch4_window7_224_22k.pth \
>> logs/cub/$tag 2>&1 & \
tail -fn 50 logs/cub/$tag
#--mask \
#--swap \
#--con \
#--margin 2 \
#--origin_w 1 \
#--swap_w 0.5 \
#--con_w 0.5 \
#--num_part 2 \
#--mask \
#--swap \
#--con \
#--margin 1 \
#--origin_w 1 \
#--swap_w 1 \
#--con_w 1 \
