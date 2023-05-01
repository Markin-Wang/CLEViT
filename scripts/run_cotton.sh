export tag='cotton_sb7_fullb_11075m1_np2_lr1e-3_mul5_bs12_448_ep200_linear'
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py \
--dataset cotton \
--tag $tag \
--lr 1e-3 \
--model full_b \
--mask \
--swap \
--con \
--margin 1 \
--origin_w 1 \
--swap_w 0.75 \
--con_w 1 \
--num_part 2 \
--img_size 448 \
--cfg configs/swin/swin_base_patch4_window7_448.yaml \
--data-path ./datasets \
--batch-size 12 \
--epochs 200 \
--pretrained ./pretrained_models/swin_base_patch4_window7_224_22k.pth \
>> logs/cotton/$tag 2>&1 & \
tail -fn 50 logs/cotton/$tag
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

