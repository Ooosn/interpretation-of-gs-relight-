date="241203"
subtask="real_scenes"
data_root="/path/to/data"
view_num=2000

## ==========================================================
## =======================LightStage=========================

python train.py -s $data_root/LightStage/Container  \
                --hdr \
                --data_device "cpu" \
                --view_num $view_num \
                --iterations 100000 \
                --asg_freeze_step 22000 \
                --spcular_freeze_step 9000 \
                --fit_linear_step 7000 \
                --asg_lr_freeze_step 40000 \
                --asg_lr_max_steps 50000 \
                --asg_lr_init 0.01 \
                --asg_lr_final 0.0001 \
                --local_q_lr_freeze_step 40000 \
                --local_q_lr_init 0.01 \
                --local_q_lr_final 0.0001 \
                --local_q_lr_max_steps 50000 \
                --neural_phasefunc_lr_init 0.001 \
                --neural_phasefunc_lr_final 0.00001 \
                --freeze_phasefunc_steps 50000 \
                --neural_phasefunc_lr_max_steps 50000 \
                --position_lr_max_steps 70000 \
                --densify_until_iter 90000 \
                --test_iterations 2000 7000 10000 15000 20000 25000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --save_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --checkpoint_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --unfreeze_iterations 5000 \
                -m "./output/$date/$subtask/Container" \
                --use_nerual_phasefunc \
                --cam_opt \
                --pl_opt \
                --densify_grad_threshold 0.0001 \
                --eval 
                
python train.py -s $data_root/LightStage/Li\'lOnes  \
                --hdr \
                --data_device "cpu" \
                --view_num $view_num \
                --iterations 100000 \
                --asg_freeze_step 22000 \
                --spcular_freeze_step 9000 \
                --fit_linear_step 7000 \
                --asg_lr_freeze_step 40000 \
                --asg_lr_max_steps 50000 \
                --asg_lr_init 0.01 \
                --asg_lr_final 0.0001 \
                --local_q_lr_freeze_step 40000 \
                --local_q_lr_init 0.01 \
                --local_q_lr_final 0.0001 \
                --local_q_lr_max_steps 50000 \
                --neural_phasefunc_lr_init 0.001 \
                --neural_phasefunc_lr_final 0.00001 \
                --freeze_phasefunc_steps 50000 \
                --neural_phasefunc_lr_max_steps 50000 \
                --position_lr_max_steps 70000 \
                --densify_until_iter 90000 \
                --test_iterations 2000 7000 10000 15000 20000 25000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --save_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --checkpoint_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --unfreeze_iterations 5000 \
                -m "./output/$date/$subtask/LilOnes" \
                --use_nerual_phasefunc \
                --cam_opt \
                --pl_opt \
                --densify_grad_threshold 0.00004 \
                --eval 
                
python train.py -s $data_root/LightStage/Boot  \
                --hdr \
                --data_device "cpu" \
                --view_num $view_num \
                --iterations 100000 \
                --asg_freeze_step 22000 \
                --spcular_freeze_step 9000 \
                --fit_linear_step 7000 \
                --asg_lr_freeze_step 40000 \
                --asg_lr_max_steps 50000 \
                --asg_lr_init 0.01 \
                --asg_lr_final 0.0001 \
                --local_q_lr_freeze_step 40000 \
                --local_q_lr_init 0.01 \
                --local_q_lr_final 0.0001 \
                --local_q_lr_max_steps 50000 \
                --neural_phasefunc_lr_init 0.001 \
                --neural_phasefunc_lr_final 0.00001 \
                --freeze_phasefunc_steps 50000 \
                --neural_phasefunc_lr_max_steps 50000 \
                --position_lr_max_steps 70000 \
                --densify_until_iter 130000 \
                --test_iterations 2000 7000 10000 15000 20000 25000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --save_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --checkpoint_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --unfreeze_iterations 5000 \
                -m "./output/$date/$subtask/Boot" \
                --use_nerual_phasefunc \
                --cam_opt \
                --pl_opt \
                --densify_grad_threshold 0.00006 \
                --eval 
                
python train.py -s $data_root/LightStage/Fox  \
                --hdr \
                --data_device "cpu" \
                --view_num $view_num \
                --iterations 100000 \
                --asg_freeze_step 22000 \
                --spcular_freeze_step 9000 \
                --fit_linear_step 7000 \
                --asg_lr_freeze_step 40000 \
                --asg_lr_max_steps 50000 \
                --asg_lr_init 0.01 \
                --asg_lr_final 0.0001 \
                --local_q_lr_freeze_step 40000 \
                --local_q_lr_init 0.01 \
                --local_q_lr_final 0.0001 \
                --local_q_lr_max_steps 50000 \
                --neural_phasefunc_lr_init 0.001 \
                --neural_phasefunc_lr_final 0.00001 \
                --freeze_phasefunc_steps 50000 \
                --neural_phasefunc_lr_max_steps 50000 \
                --position_lr_max_steps 70000 \
                --densify_until_iter 90000 \
                --test_iterations 2000 7000 10000 15000 20000 25000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --save_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --checkpoint_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --unfreeze_iterations 5000 \
                -m "./output/$date/$subtask/Fox" \
                --use_nerual_phasefunc \
                --cam_opt \
                --pl_opt \
                --densify_grad_threshold 0.0001 \
                --eval 
                
python train.py -s $data_root/LightStage/Nefertiti  \
                --hdr \
                --data_device "cpu" \
                --view_num $view_num \
                --iterations 100000 \
                --asg_freeze_step 22000 \
                --spcular_freeze_step 9000 \
                --fit_linear_step 7000 \
                --asg_lr_freeze_step 40000 \
                --asg_lr_max_steps 50000 \
                --asg_lr_init 0.01 \
                --asg_lr_final 0.0001 \
                --local_q_lr_freeze_step 40000 \
                --local_q_lr_init 0.01 \
                --local_q_lr_final 0.0001 \
                --local_q_lr_max_steps 50000 \
                --neural_phasefunc_lr_init 0.001 \
                --neural_phasefunc_lr_final 0.00001 \
                --freeze_phasefunc_steps 50000 \
                --neural_phasefunc_lr_max_steps 50000 \
                --position_lr_max_steps 70000 \
                --densify_until_iter 90000 \
                --test_iterations 2000 7000 10000 15000 20000 25000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --save_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --checkpoint_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --unfreeze_iterations 5000 \
                -m "./output/$date/$subtask/Nefertiti" \
                --use_nerual_phasefunc \
                --cam_opt \
                --pl_opt \
                --densify_grad_threshold 0.0001 \
                --eval 
                
python train.py -s $data_root/LightStage/Zhaojun  \
                --hdr \
                --data_device "cpu" \
                --view_num $view_num \
                --iterations 100000 \
                --asg_freeze_step 22000 \
                --spcular_freeze_step 9000 \
                --fit_linear_step 7000 \
                --asg_lr_freeze_step 40000 \
                --asg_lr_max_steps 50000 \
                --asg_lr_init 0.01 \
                --asg_lr_final 0.0001 \
                --local_q_lr_freeze_step 40000 \
                --local_q_lr_init 0.01 \
                --local_q_lr_final 0.0001 \
                --local_q_lr_max_steps 50000 \
                --neural_phasefunc_lr_init 0.001 \
                --neural_phasefunc_lr_final 0.00001 \
                --freeze_phasefunc_steps 50000 \
                --neural_phasefunc_lr_max_steps 50000 \
                --position_lr_max_steps 70000 \
                --densify_until_iter 130000 \
                --test_iterations 2000 7000 10000 15000 20000 25000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --save_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --checkpoint_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --unfreeze_iterations 5000 \
                -m "./output/$date/$subtask/Zhaojun" \
                --use_nerual_phasefunc \
                --cam_opt \
                --pl_opt \
                --densify_grad_threshold 0.00006 \
                --eval 

## ==========================================================
## =======================NRHints============================
                
python train.py -s $data_root/NRHints/Pikachu  \
                --data_device "cpu" \
                --view_num $view_num \
                --iterations 100000 \
                --asg_freeze_step 22000 \
                --spcular_freeze_step 9000 \
                --fit_linear_step 7000 \
                --asg_lr_freeze_step 40000 \
                --asg_lr_max_steps 50000 \
                --asg_lr_init 0.01 \
                --asg_lr_final 0.0001 \
                --local_q_lr_freeze_step 40000 \
                --local_q_lr_init 0.01 \
                --local_q_lr_final 0.0001 \
                --local_q_lr_max_steps 50000 \
                --neural_phasefunc_lr_init 0.001 \
                --neural_phasefunc_lr_final 0.00001 \
                --freeze_phasefunc_steps 50000 \
                --neural_phasefunc_lr_max_steps 50000 \
                --position_lr_max_steps 70000 \
                --densify_until_iter 90000 \
                --test_iterations 2000 7000 10000 15000 20000 25000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --save_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --checkpoint_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --unfreeze_iterations 5000 \
                -m "./output/$date/$subtask/Pikachu" \
                --use_nerual_phasefunc \
                --cam_opt \
                --pl_opt \
                --densify_grad_threshold 0.00015 \
                --eval 
                
python train.py -s $data_root/NRHints/Cluttered  \
                --data_device "cpu" \
                --view_num $view_num \
                --iterations 100000 \
                --asg_freeze_step 22000 \
                --spcular_freeze_step 9000 \
                --fit_linear_step 7000 \
                --asg_lr_freeze_step 40000 \
                --asg_lr_max_steps 50000 \
                --asg_lr_init 0.01 \
                --asg_lr_final 0.0001 \
                --local_q_lr_freeze_step 40000 \
                --local_q_lr_init 0.01 \
                --local_q_lr_final 0.0001 \
                --local_q_lr_max_steps 50000 \
                --neural_phasefunc_lr_init 0.001 \
                --neural_phasefunc_lr_final 0.00001 \
                --freeze_phasefunc_steps 50000 \
                --neural_phasefunc_lr_max_steps 50000 \
                --position_lr_max_steps 70000 \
                --densify_until_iter 90000 \
                --test_iterations 2000 7000 10000 15000 20000 25000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --save_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --checkpoint_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --unfreeze_iterations 5000 \
                -m "./output/$date/$subtask/Cluttered" \
                --use_nerual_phasefunc \
                --cam_opt \
                --pl_opt \
                --densify_grad_threshold 0.0002 \
                --eval 
                
python train.py -s $data_root/NRHints/Cup-Fabric  \
                --data_device "cpu" \
                --view_num $view_num \
                --iterations 100000 \
                --asg_freeze_step 22000 \
                --spcular_freeze_step 9000 \
                --fit_linear_step 7000 \
                --asg_lr_freeze_step 40000 \
                --asg_lr_max_steps 50000 \
                --asg_lr_init 0.01 \
                --asg_lr_final 0.0001 \
                --local_q_lr_freeze_step 40000 \
                --local_q_lr_init 0.01 \
                --local_q_lr_final 0.0001 \
                --local_q_lr_max_steps 50000 \
                --neural_phasefunc_lr_init 0.001 \
                --neural_phasefunc_lr_final 0.00001 \
                --freeze_phasefunc_steps 50000 \
                --neural_phasefunc_lr_max_steps 50000 \
                --position_lr_max_steps 70000 \
                --densify_until_iter 90000 \
                --test_iterations 2000 7000 10000 15000 20000 25000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --save_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --checkpoint_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --unfreeze_iterations 5000 \
                -m "./output/$date/$subtask/Cup-Fabric" \
                --use_nerual_phasefunc \
                --cam_opt \
                --pl_opt \
                --densify_grad_threshold 0.0001 \
                --eval 

python train.py -s $data_root/NRHints/Fish  \
                --data_device "cpu" \
                --view_num $view_num \
                --iterations 100000 \
                --asg_freeze_step 22000 \
                --spcular_freeze_step 9000 \
                --fit_linear_step 7000 \
                --asg_lr_freeze_step 40000 \
                --asg_lr_max_steps 50000 \
                --asg_lr_init 0.01 \
                --asg_lr_final 0.0001 \
                --local_q_lr_freeze_step 40000 \
                --local_q_lr_init 0.01 \
                --local_q_lr_final 0.0001 \
                --local_q_lr_max_steps 50000 \
                --neural_phasefunc_lr_init 0.001 \
                --neural_phasefunc_lr_final 0.00001 \
                --freeze_phasefunc_steps 50000 \
                --neural_phasefunc_lr_max_steps 50000 \
                --position_lr_max_steps 70000 \
                --densify_until_iter 100000 \
                --test_iterations 2000 7000 10000 15000 20000 25000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --save_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --checkpoint_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --unfreeze_iterations 5000 \
                -m "./output/$date/$subtask/Fish" \
                --use_nerual_phasefunc \
                --cam_opt \
                --pl_opt \
                --densify_grad_threshold 0.00012 \
                --eval 
                
python train.py -s $data_root/NRHints/Cat  \
                --data_device "cpu" \
                --view_num $view_num \
                --iterations 100000 \
                --asg_freeze_step 22000 \
                --spcular_freeze_step 9000 \
                --fit_linear_step 7000 \
                --asg_lr_freeze_step 40000 \
                --asg_lr_max_steps 50000 \
                --asg_lr_init 0.01 \
                --asg_lr_final 0.0001 \
                --local_q_lr_freeze_step 40000 \
                --local_q_lr_init 0.01 \
                --local_q_lr_final 0.0001 \
                --local_q_lr_max_steps 50000 \
                --neural_phasefunc_lr_init 0.001 \
                --neural_phasefunc_lr_final 0.00001 \
                --freeze_phasefunc_steps 50000 \
                --neural_phasefunc_lr_max_steps 50000 \
                --position_lr_max_steps 70000 \
                --densify_until_iter 90000 \
                --test_iterations 2000 7000 10000 15000 20000 25000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --save_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --checkpoint_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --unfreeze_iterations 5000 \
                -m "./output/$date/$subtask/Cat" \
                --use_nerual_phasefunc \
                --cam_opt \
                --pl_opt \
                --densify_grad_threshold 0.0002 \
                --eval 
                
python train.py -s $data_root/NRHints/Pixiu  \
                --data_device "cpu" \
                --view_num $view_num \
                --iterations 100000 \
                --asg_freeze_step 22000 \
                --spcular_freeze_step 9000 \
                --fit_linear_step 7000 \
                --asg_lr_freeze_step 40000 \
                --asg_lr_max_steps 50000 \
                --asg_lr_init 0.01 \
                --asg_lr_final 0.0001 \
                --local_q_lr_freeze_step 40000 \
                --local_q_lr_init 0.01 \
                --local_q_lr_final 0.0001 \
                --local_q_lr_max_steps 50000 \
                --neural_phasefunc_lr_init 0.001 \
                --neural_phasefunc_lr_final 0.00001 \
                --freeze_phasefunc_steps 50000 \
                --neural_phasefunc_lr_max_steps 50000 \
                --position_lr_max_steps 70000 \
                --densify_until_iter 90000 \
                --test_iterations 2000 7000 10000 15000 20000 25000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --save_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --checkpoint_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --unfreeze_iterations 5000 \
                -m "./output/$date/$subtask/Pixiu" \
                --use_nerual_phasefunc \
                --cam_opt \
                --pl_opt \
                --densify_grad_threshold 0.0002 \
                --eval 
                
python train.py -s $data_root/NRHints/Cat_on_Decor  \
                --data_device "cpu" \
                --view_num $view_num \
                --iterations 100000 \
                --asg_freeze_step 22000 \
                --spcular_freeze_step 9000 \
                --fit_linear_step 7000 \
                --asg_lr_freeze_step 40000 \
                --asg_lr_max_steps 50000 \
                --asg_lr_init 0.01 \
                --asg_lr_final 0.0001 \
                --local_q_lr_freeze_step 40000 \
                --local_q_lr_init 0.01 \
                --local_q_lr_final 0.0001 \
                --local_q_lr_max_steps 50000 \
                --neural_phasefunc_lr_init 0.001 \
                --neural_phasefunc_lr_final 0.00001 \
                --freeze_phasefunc_steps 50000 \
                --neural_phasefunc_lr_max_steps 50000 \
                --position_lr_max_steps 70000 \
                --densify_until_iter 90000 \
                --test_iterations 2000 7000 10000 15000 20000 25000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --save_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --checkpoint_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --unfreeze_iterations 5000 \
                -m "./output/$date/$subtask/Cat_on_Decor" \
                --use_nerual_phasefunc \
                --cam_opt \
                --pl_opt \
                --densify_grad_threshold 0.0002 \
                --eval 
