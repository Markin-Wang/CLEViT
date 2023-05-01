export tag='afd_sb7_fullb_111m1_np2_lr1e-3_mul5_bs12_448_ep100_linear_oricon'
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py \
--dataset AFD \
--tag $tag \
--lr 1e-3 \
--model full_b \
--mask \
--swap \
--con \
--margin 1 \
--origin_w 1 \
--swap_w 1 \
--con_w 1 \
--num_part 2 \
--img_size 448 \
--cfg configs/swin/swin_base_patch4_window7_448.yaml \
--data-path ./datasets \
--batch-size 12 \
--eval-batch-size 36 \
--epochs 100 \
--pretrained ./pretrained_models/swin_base_patch4_window7_224_22k.pth \
>> logs/afd/$tag 2>&1 & \
tail -fn 50 logs/afd/$tag
#--con_w 1 \
#--con \
#--mask \
#--swap \
#--con \
#--margin 1 \
#--origin_w 1 \
#--swap_w 1 \
#--con_w 1 \
#--num_part 2 \

