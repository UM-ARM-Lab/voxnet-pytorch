python train.py \
    --rope_data \
    --model voxnet_multientry \
    --log_dir log_10 \
    --num_classes 1 \
    --max_epoch 1 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --momentum 0.9 \
    --optimizer adam \
    --decay_step 4 \
    --decay_rate 0.8 \
    --saved_fname rope_multientry_same_net \
    --num_channels 3 \
    --save_to_pt \
    --use_same_net
    #--cont \
    #--ckpt_dir \
    #--ckpt_fname 
