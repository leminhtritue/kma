python image_target.py --nrf 16384 --s 2 --t 1 --output_src ckps/source2_16384_gm0_1_aw0_1_vat0_arf0_1_ep500/ --grid 1.0 --gpu_id 0 2>&1 | tee outfile_0 #4
python image_target.py --nrf 16384 --s 2 --t 1 --output_src ckps/source2_16384_gm0_1_aw0_1_vat0_arf1_ep500/ --grid 1.0 --gpu_id 1 2>&1 | tee outfile_1
python image_target.py --nrf 16384 --s 2 --t 1 --output_src ckps/source2_16384_gm0_1_aw0_1_vat1_arf0_1_ep500/ --grid 1.0 --gpu_id 2 2>&1 | tee outfile_2
python image_target.py --nrf 16384 --s 2 --t 1 --output_src ckps/source2_16384_gm0_1_aw0_1_vat1_arf1_ep500/ --grid 1.0 --gpu_id 3 2>&1 | tee outfile_3 



python image_target.py --nrf 16384 --s 0 --t 1 --output_src ckps/source0_16384_gm0_1_aw0_1_vat1_arf0_1_ep500/ --w_vat 0.0 --cls_par 0.3 --alpha_rfen 1.0 --alpha_rf 0.0 --max_zero 0.0 --gpu_id 0
python image_target.py --nrf 16384 --s 0 --t 2 --output_src ckps/source0_16384_gm0_1_aw0_1_vat1_arf0_1_ep500/ --w_vat 0.0 --cls_par 0.3 --alpha_rfen 1.0 --alpha_rf 0.0 --max_zero 0.0 --gpu_id 1
python image_target.py --nrf 16384 --s 1 --t 0 --output_src ckps/source1_16384_gm0_1_aw0_1_vat1_arf0_1_ep500/ --w_vat 0.0 --cls_par 0.3 --alpha_rfen 1.0 --alpha_rf 0.0 --max_zero 0.0 --gpu_id 2
python image_target.py --nrf 16384 --s 1 --t 2 --output_src ckps/source1_16384_gm0_1_aw0_1_vat1_arf0_1_ep500/ --w_vat 0.0 --cls_par 0.3 --alpha_rfen 1.0 --alpha_rf 0.0 --max_zero 0.0 --gpu_id 3
python image_target.py --nrf 16384 --s 2 --t 0 --output_src ckps/source2_16384_gm0_1_aw0_1_vat1_arf0_1_ep500/ --w_vat 0.0 --cls_par 0.3 --alpha_rfen 1.0 --alpha_rf 0.0 --max_zero 0.0 --gpu_id 0 #4
python image_target.py --nrf 16384 --s 2 --t 1 --output_src ckps/source2_16384_gm0_1_aw0_1_vat1_arf0_1_ep500/ --w_vat 0.0 --cls_par 0.3 --alpha_rfen 1.0 --alpha_rf 0.0 --max_zero 0.0 --gpu_id 0

python image_target.py --nrf 16384 --s 0 --t 1 --output_src ckps/source0_16384_gm0_1_aw0_1_vat1_arf0_1_ep500/ --w_vat 0.1 --cls_par 0.3 --alpha_rfen 1.0 --alpha_rf 0.0 --max_zero 0.0 --gpu_id 1
python image_target.py --nrf 16384 --s 0 --t 2 --output_src ckps/source0_16384_gm0_1_aw0_1_vat1_arf0_1_ep500/ --w_vat 0.1 --cls_par 0.3 --alpha_rfen 1.0 --alpha_rf 0.0 --max_zero 0.0 --gpu_id 1
python image_target.py --nrf 16384 --s 1 --t 0 --output_src ckps/source1_16384_gm0_1_aw0_1_vat1_arf0_1_ep500/ --w_vat 0.1 --cls_par 0.3 --alpha_rfen 1.0 --alpha_rf 0.0 --max_zero 0.0 --gpu_id 1
python image_target.py --nrf 16384 --s 1 --t 2 --output_src ckps/source1_16384_gm0_1_aw0_1_vat1_arf0_1_ep500/ --w_vat 0.1 --cls_par 0.3 --alpha_rfen 1.0 --alpha_rf 0.0 --max_zero 0.0 --gpu_id 1
python image_target.py --nrf 16384 --s 2 --t 0 --output_src ckps/source2_16384_gm0_1_aw0_1_vat1_arf0_1_ep500/ --w_vat 0.1 --cls_par 0.3 --alpha_rfen 1.0 --alpha_rf 0.0 --max_zero 0.0 --gpu_id 1
python image_target.py --nrf 16384 --s 2 --t 1 --output_src ckps/source2_16384_gm0_1_aw0_1_vat1_arf0_1_ep500/ --w_vat 0.1 --cls_par 0.3 --alpha_rfen 1.0 --alpha_rf 0.0 --max_zero 0.0 --gpu_id 1

python image_target.py --nrf 16384 --s 0 --t 1 --output_src ckps/source0_16384_gm0_1_aw0_1_vat1_arf0_1_ep500/ --w_vat 0.1 --cls_par 0.3 --alpha_rfen 0.0 --alpha_rf 0.0 --max_zero 0.0 --gpu_id 2
python image_target.py --nrf 16384 --s 0 --t 2 --output_src ckps/source0_16384_gm0_1_aw0_1_vat1_arf0_1_ep500/ --w_vat 0.1 --cls_par 0.3 --alpha_rfen 0.0 --alpha_rf 0.0 --max_zero 0.0 --gpu_id 2
python image_target.py --nrf 16384 --s 1 --t 0 --output_src ckps/source1_16384_gm0_1_aw0_1_vat1_arf0_1_ep500/ --w_vat 0.1 --cls_par 0.3 --alpha_rfen 0.0 --alpha_rf 0.0 --max_zero 0.0 --gpu_id 2
python image_target.py --nrf 16384 --s 1 --t 2 --output_src ckps/source1_16384_gm0_1_aw0_1_vat1_arf0_1_ep500/ --w_vat 0.1 --cls_par 0.3 --alpha_rfen 0.0 --alpha_rf 0.0 --max_zero 0.0 --gpu_id 2
python image_target.py --nrf 16384 --s 2 --t 0 --output_src ckps/source2_16384_gm0_1_aw0_1_vat1_arf0_1_ep500/ --w_vat 0.1 --cls_par 0.3 --alpha_rfen 0.0 --alpha_rf 0.0 --max_zero 0.0 --gpu_id 2
python image_target.py --nrf 16384 --s 2 --t 1 --output_src ckps/source2_16384_gm0_1_aw0_1_vat1_arf0_1_ep500/ --w_vat 0.1 --cls_par 0.3 --alpha_rfen 0.0 --alpha_rf 0.0 --max_zero 0.0 --gpu_id 2

python image_target.py --nrf 16384 --s 0 --t 1 --output_src ckps/source0_16384_gm0_1_aw0_1_vat1_arf0_1_ep500/ --w_vat 0.0 --cls_par 0.3 --alpha_rfen 0.0 --alpha_rf 0.0 --max_zero 0.0 --gpu_id 3
python image_target.py --nrf 16384 --s 0 --t 2 --output_src ckps/source0_16384_gm0_1_aw0_1_vat1_arf0_1_ep500/ --w_vat 0.0 --cls_par 0.3 --alpha_rfen 0.0 --alpha_rf 0.0 --max_zero 0.0 --gpu_id 3
python image_target.py --nrf 16384 --s 1 --t 0 --output_src ckps/source1_16384_gm0_1_aw0_1_vat1_arf0_1_ep500/ --w_vat 0.0 --cls_par 0.3 --alpha_rfen 0.0 --alpha_rf 0.0 --max_zero 0.0 --gpu_id 3
python image_target.py --nrf 16384 --s 1 --t 2 --output_src ckps/source1_16384_gm0_1_aw0_1_vat1_arf0_1_ep500/ --w_vat 0.0 --cls_par 0.3 --alpha_rfen 0.0 --alpha_rf 0.0 --max_zero 0.0 --gpu_id 3
python image_target.py --nrf 16384 --s 2 --t 0 --output_src ckps/source2_16384_gm0_1_aw0_1_vat1_arf0_1_ep500/ --w_vat 0.0 --cls_par 0.3 --alpha_rfen 0.0 --alpha_rf 0.0 --max_zero 0.0 --gpu_id 3
python image_target.py --nrf 16384 --s 2 --t 1 --output_src ckps/source2_16384_gm0_1_aw0_1_vat1_arf0_1_ep500/ --w_vat 0.0 --cls_par 0.3 --alpha_rfen 0.0 --alpha_rf 0.0 --max_zero 0.0 --gpu_id 3

python image_target.py --nrf 16384 --s 0 --t 1 --output_src ckps/source0_16384_gm0_1_aw0_1_vat1_arf0_1_ep500/ --w_vat 1.0 --cls_par 0.3 --alpha_rfen 1.0 --alpha_rf 0.0 --max_zero 0.0 --gpu_id 1
python image_target.py --nrf 16384 --s 0 --t 2 --output_src ckps/source0_16384_gm0_1_aw0_1_vat1_arf0_1_ep500/ --w_vat 1.0 --cls_par 0.3 --alpha_rfen 1.0 --alpha_rf 0.0 --max_zero 0.0 --gpu_id 0
python image_target.py --nrf 16384 --s 1 --t 0 --output_src ckps/source1_16384_gm0_1_aw0_1_vat1_arf0_1_ep500/ --w_vat 1.0 --cls_par 0.3 --alpha_rfen 1.0 --alpha_rf 0.0 --max_zero 0.0 --gpu_id 2
python image_target.py --nrf 16384 --s 1 --t 2 --output_src ckps/source1_16384_gm0_1_aw0_1_vat1_arf0_1_ep500/ --w_vat 1.0 --cls_par 0.3 --alpha_rfen 1.0 --alpha_rf 0.0 --max_zero 0.0 --gpu_id 3
python image_target.py --nrf 16384 --s 2 --t 0 --output_src ckps/source2_16384_gm0_1_aw0_1_vat1_arf0_1_ep500/ --w_vat 1.0 --cls_par 0.3 --alpha_rfen 1.0 --alpha_rf 0.0 --max_zero 0.0 --gpu_id 0 
python image_target.py --nrf 16384 --s 2 --t 1 --output_src ckps/source2_16384_gm0_1_aw0_1_vat1_arf0_1_ep500/ --w_vat 1.0 --cls_par 0.3 --alpha_rfen 1.0 --alpha_rf 0.0 --max_zero 0.0 --gpu_id 0