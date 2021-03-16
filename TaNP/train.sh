first_embedding_dim=32
second_embedding_dim=16
z1_dim=32
z2_dim=32
z_dim=32
enc_h1_dim=32
enc_h2_dim=16
taskenc_h1_dim=32
taskenc_h2_dim=32
taskenc_final_dim=16
clusters_k=10
temperature=1.0
lambda=1.0
dec_h1_dim=32
dec_h2_dim=32
dec_h3_dim=16
dropout_rate=0
lr=0.0001
optim='adam'
num_epoch=100
batch_size=32
train_ratio=0.7
valid_ratio=0.1
support_size=20
query_size=10
max_len=200
context_min=20
CUDA_VISIBLE_DEVICES=0 python train_TaNP.py \
 --first_embedding_dim $first_embedding_dim \
 --second_embedding_dim $second_embedding_dim \
 --z1_dim $z1_dim \
 --z2_dim $z2_dim \
 --z_dim $z_dim \
 --enc_h1_dim $enc_h1_dim \
 --enc_h2_dim $enc_h2_dim \
 --taskenc_h1_dim $taskenc_h1_dim \
 --taskenc_h2_dim $taskenc_h2_dim \
 --taskenc_final_dim $taskenc_final_dim \
 --clusters_k $clusters_k \
 --lambda $lambda \
 --temperature $temperature \
 --dec_h1_dim $dec_h1_dim \
 --dec_h2_dim $dec_h2_dim \
 --dec_h3_dim $dec_h3_dim \
 --lr $lr \
 --dropout_rate $dropout_rate \
 --optim $optim \
 --num_epoch $num_epoch \
 --batch_size $batch_size \
 --train_ratio $train_ratio \
 --valid_ratio $valid_ratio \
 --support_size $support_size \
 --query_size $query_size \
 --max_len $max_len \
 --context_min $context_min
