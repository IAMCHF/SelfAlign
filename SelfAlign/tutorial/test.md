export PYTHONPATH=/HBV/Caohaofan/selfalign:$PATH 
(align)
python ./SelfAlign/bin/selfalign.py prepare_subtomo_star \
            /HBV/Caohaofan/simulation_degree30 \
            --output_star subtomo.star

python ./SelfAlign/bin/selfalign.py refine --subtomo_star subtomo.star \
                    --epochs 1 \
                    --iterations 100 \
                    --subtomo_size 40 \
                    --gpuID 1 \
                    --batch_size 64 \
                    --result_dir /HBV/Caohaofan/results \
                    --rota /HBV/Caohaofan/rota \
                    --preprocessing_ncpus 22 \
                    --select_subtomo_number 72



"5LQW, 5MPA, 5T2C, 6A5L"
python ./SelfAlign/bin/selfalign.py prepare_subtomo_star \
            /newdata3/chf/normalized/snr01/6A5L \
            --output_star 6A5L.star
python ./SelfAlign/bin/selfalign.py refine --subtomo_star 6A5L.star \
                    --epochs 1 \
                    --iterations 1 \
                    --subtomo_size 32 \
                    --gpuID 1 \
                    --batch_size 1 \
                    --result_dir /newdata3/chf/test_data2/6A5L \
                    --rota /newdata3/chf/test_data2/rota/6A5L \
                    --preprocessing_ncpus 1
python ./SelfAlign/preprocessing/folder_apply_wedge.py


python ./SelfAlign/bin/selfalign.py prepare_subtomo_star \
            /HBV/Caohaofan/subtomos_bin5_globally_normalized \
            --output_star subtomo.star

python ./SelfAlign/bin/selfalign.py refine --subtomo_star subtomo.star \
                    --epochs 1 \
                    --iterations 1 \
                    --subtomo_size 40 \
                    --gpuID 1 \
                    --batch_size 64 \
                    --result_dir /HBV/Caohaofan/results_smi \
                    --rota /HBV/Caohaofan/simulation_euler \
                    --select_subtomo_number 128 \
                    --preprocessing_ncpus 22

python ./SelfAlign/models/gumnet/train_demo.py

python ./SelfAlign/models/gumnet/test_demo.py

python ./SelfAlign/preprocessing/eval.py

python ./SelfAlign/preprocessing/folder_apply_wedge.py

python ./SelfAlign/preprocessing/snr.py

python ./normalize_z_score_foler.py