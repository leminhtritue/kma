python image_source.py --nrf 16384 --w_vat 0.0 --alpha_rf 1.0 --max_epoch 30 --s 0 --t 1 --output ckps/source0_16384_gm0_1_aw0_1_vat0_arf1_ep30_run2/ --gpu_id 0
python image_target.py --nrf 16384 --s 0 --t 1 --output_src ckps/source0_16384_gm0_1_aw0_1_vat0_arf1_ep30_run2/ --w_vat 1.0 --cls_par 0.0 --alpha_rf 0.1 --max_zero 0.0 --gpu_id 0
python image_target.py --nrf 16384 --s 0 --t 1 --output_src ckps/source0_16384_gm0_1_aw0_1_vat0_arf1_ep30_run2/ --w_vat 1.0 --cls_par 0.1 --alpha_rf 0.1 --max_zero 0.0 --gpu_id 1
python image_target.py --nrf 16384 --s 0 --t 2 --output_src ckps/source0_16384_gm0_1_aw0_1_vat0_arf1_ep30_run2/ --w_vat 1.0 --cls_par 0.0 --alpha_rf 0.1 --max_zero 0.0 --gpu_id 2
python image_target.py --nrf 16384 --s 0 --t 2 --output_src ckps/source0_16384_gm0_1_aw0_1_vat0_arf1_ep30_run2/ --w_vat 1.0 --cls_par 0.1 --alpha_rf 0.1 --max_zero 0.0 --gpu_id 3

python image_source.py --nrf 16384 --w_vat 0.0 --alpha_rf 1.0 --max_epoch 30 --s 1 --t 0 --output ckps/source1_16384_gm0_1_aw0_1_vat0_arf1_ep30/ --gpu_id 1
python image_target.py --nrf 16384 --s 1 --t 0 --output_src ckps/source1_16384_gm0_1_aw0_1_vat0_arf1_ep30/ --w_vat 1.0 --cls_par 0.0 --alpha_rf 0.1 --max_zero 0.0 --gpu_id 0
python image_target.py --nrf 16384 --s 1 --t 0 --output_src ckps/source1_16384_gm0_1_aw0_1_vat0_arf1_ep30/ --w_vat 1.0 --cls_par 0.1 --alpha_rf 0.1 --max_zero 0.0 --gpu_id 1
python image_target.py --nrf 16384 --s 1 --t 2 --output_src ckps/source1_16384_gm0_1_aw0_1_vat0_arf1_ep30/ --w_vat 1.0 --cls_par 0.0 --alpha_rf 0.1 --max_zero 0.0 --gpu_id 2
python image_target.py --nrf 16384 --s 1 --t 2 --output_src ckps/source1_16384_gm0_1_aw0_1_vat0_arf1_ep30/ --w_vat 1.0 --cls_par 0.1 --alpha_rf 0.1 --max_zero 0.0 --gpu_id 3

python image_source.py --nrf 16384 --w_vat 0.0 --alpha_rf 1.0 --max_epoch 30 --s 2 --t 0 --output ckps/source2_16384_gm0_1_aw0_1_vat0_arf1_ep30/ --gpu_id 2
python image_target.py --nrf 16384 --s 2 --t 0 --output_src ckps/source2_16384_gm0_1_aw0_1_vat0_arf1_ep30/ --w_vat 1.0 --cls_par 0.0 --alpha_rf 0.1 --max_zero 0.0 --gpu_id 0
python image_target.py --nrf 16384 --s 2 --t 0 --output_src ckps/source2_16384_gm0_1_aw0_1_vat0_arf1_ep30/ --w_vat 1.0 --cls_par 0.1 --alpha_rf 0.1 --max_zero 0.0 --gpu_id 1
python image_target.py --nrf 16384 --s 2 --t 1 --output_src ckps/source2_16384_gm0_1_aw0_1_vat0_arf1_ep30/ --w_vat 1.0 --cls_par 0.0 --alpha_rf 0.1 --max_zero 0.0 --gpu_id 2
python image_target.py --nrf 16384 --s 2 --t 1 --output_src ckps/source2_16384_gm0_1_aw0_1_vat0_arf1_ep30/ --w_vat 1.0 --cls_par 0.1 --alpha_rf 0.1 --max_zero 0.0 --gpu_id 3






High
python image_source.py --nrf 16384 --w_vat 0.0 --alpha_rf 1.0 --max_epoch 500 --s 0 --t 1 --output ckps/source0_16384_gm0_1_aw0_1_vat0_arf1_ep500/ --gpu_id 1
python image_target.py --nrf 16384 --s 0 --t 1 --output_src ckps/source0_16384_gm0_1_aw0_1_vat0_arf1_ep500/ --w_vat 1.0 --cls_par 0.0 --alpha_rf 0.1 --max_zero 0.0 --gpu_id 0
python image_target.py --nrf 16384 --s 0 --t 1 --output_src ckps/source0_16384_gm0_1_aw0_1_vat0_arf1_ep500/ --w_vat 1.0 --cls_par 0.1 --alpha_rf 0.1 --max_zero 0.0 --gpu_id 1
python image_target.py --nrf 16384 --s 0 --t 2 --output_src ckps/source0_16384_gm0_1_aw0_1_vat0_arf1_ep500/ --w_vat 1.0 --cls_par 0.0 --alpha_rf 0.1 --max_zero 0.0 --gpu_id 2
python image_target.py --nrf 16384 --s 0 --t 2 --output_src ckps/source0_16384_gm0_1_aw0_1_vat0_arf1_ep500/ --w_vat 1.0 --cls_par 0.1 --alpha_rf 0.1 --max_zero 0.0 --gpu_id 3