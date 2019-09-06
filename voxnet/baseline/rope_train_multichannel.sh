python train.py \
    --rope_data \
    --model voxnet \
    --log_dir log_10 \
    --num_classes 1 \
    --max_epoch 32 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --momentum 0.9 \
    --optimizer adam \
    --decay_step 4 \
    --decay_rate 0.8 \
    --saved_fname weight10 \
    --num_channels 3 \
    --save_to_pt
    #--cont \
    #--ckpt_dir \
    #--ckpt_fname 
