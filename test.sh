python main_dp.py --checkpoint_path "./model/axial_mm.tar" --phase="play" --vid_dir="/home/kwon/Conference/ECCV2024_axial/datasets/baby" --alpha_x 20 --alpha_y 20 --theta 0 --is_single_gpu_trained   
python main_dp.py --checkpoint_path "./model/axial_mm.tar" --phase="play" --vid_dir="/home/kwon/Conference/ECCV2024_axial/datasets/baby" --alpha_x 0 --alpha_y 20 --theta 0 --is_single_gpu_trained   
python main_dp.py --checkpoint_path "./model/axial_mm.tar" --phase="play" --vid_dir="/home/kwon/Conference/ECCV2024_axial/datasets/baby" --alpha_x 20 --alpha_y 0 --theta 0 --is_single_gpu_trained   
python main_dp.py --checkpoint_path "./model/axial_mm.tar" --phase="play" --vid_dir="/home/kwon/Conference/ECCV2024_axial/datasets/baby" --alpha_x 20 --alpha_y 0 --theta 90 --is_single_gpu_trained   
