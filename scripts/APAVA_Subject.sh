# Subject-based
# APAVA Dataset with MedMamba (DiffMamba)
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/APAVA/ \
  --model_id APAVA-MedMamba \
  --model MedMamba \
  --data APAVA \
  --e_layers 4 \
  --batch_size 64 \
  --d_model 256 \
  --d_ff 512 \
  --d_state 16 \
  --d_conv 4 \
  --expand 2 \
  --augmentations none,drop0.35 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --patience 3 \
  --dropout 0.1 \
  --num_workers 0