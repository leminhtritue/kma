python image_target.py --nrf 16384 --s 1 --t 0 --output_src ckps/source1_16384_gm0_1_aw0_1_vat1_arf0_1_ep500/ --train_step2 0.0 --lr_decayo_2 0.1 --temp2 0.0 --w_vat_2 0.0 --layer_2 wn --gpu_id 0

python image_target.py --nrf 16384 --s 1 --t 0 --output_src ckps/source1_16384_gm0_1_aw0_1_vat1_arf0_1_ep500/ --train_step2 0.0 --lr_decayo_2 0.1 --temp2 0.0 --w_vat_2 0.1 --layer_2 wn --gpu_id 1 #VAT
python image_target.py --nrf 16384 --s 1 --t 0 --output_src ckps/source1_16384_gm0_1_aw0_1_vat1_arf0_1_ep500/ --train_step2 0.0 --lr_decayo_2 0.1 --temp2 0.0 --w_vat_2 0.5 --layer_2 wn --gpu_id 2
python image_target.py --nrf 16384 --s 1 --t 0 --output_src ckps/source1_16384_gm0_1_aw0_1_vat1_arf0_1_ep500/ --train_step2 0.0 --lr_decayo_2 0.1 --temp2 0.0 --w_vat_2 1.0 --layer_2 wn --gpu_id 3 #VAT

python image_target.py --nrf 16384 --s 1 --t 0 --output_src ckps/source1_16384_gm0_1_aw0_1_vat1_arf0_1_ep500/ --train_step2 0.0 --lr_decayo_2 0.1 --temp2 1.0 --w_vat_2 0.0 --layer_2 wn --gpu_id 0 #4Temp

python image_target.py --nrf 16384 --s 1 --t 0 --output_src ckps/source1_16384_gm0_1_aw0_1_vat1_arf0_1_ep500/ --train_step2 0.0 --lr_decayo_2 0.1 --temp2 0.1 --w_vat_2 0.0 --layer_2 wn --gpu_id 0 #Temp
python image_target.py --nrf 16384 --s 1 --t 0 --output_src ckps/source1_16384_gm0_1_aw0_1_vat1_arf0_1_ep500/ --train_step2 0.0 --lr_decayo_2 0.1 --temp2 10.0 --w_vat_2 0.0 --layer_2 wn --gpu_id 1 #Temp

python image_target.py --nrf 16384 --s 1 --t 0 --output_src ckps/source1_16384_gm0_1_aw0_1_vat1_arf0_1_ep500/ --train_step2 0.1 --lr_decayo_2 0.1 --temp2 0.0 --w_vat_2 0.0 --layer_2 wn --gpu_id 2 #Trainall
python image_target.py --nrf 16384 --s 1 --t 0 --output_src ckps/source1_16384_gm0_1_aw0_1_vat1_arf0_1_ep500/ --train_step2 0.1 --lr_decayo_2 0.1 --temp2 0.0 --w_vat_2 0. --layer_2 wn --gpu_id 2 #Trainall