export tag='cotton_test'
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  extract_features.py \
--dataset soybean2000 \
--tag $tag \
--lr 1e-3 \
--model full_b \
--mask \
--swap \
--margin 1 \
--origin_w 1 \
--swap_w 1 \
--num_part 2 \
--img_size 448 \
--cfg configs/swin/swin_base_patch4_window7_448.yaml \
--data-path ./datasets \
--batch-size 1 \
--eval-batch-size 1 \
--epochs 200 \
--pretrained ./output/trained_models/soygbl/our.pth \
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

