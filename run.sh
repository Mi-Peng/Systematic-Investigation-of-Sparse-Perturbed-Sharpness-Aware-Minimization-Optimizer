python train.py \
  --model vit_nodo --dataset CIFAR100_base --datadir .. \
  --opt sam-adam --lr 1e-4 --weight_decay 0 --rho 0.2 \ 
  --seed 1234 --batch_size 16 --epochs 10 \
  --n_structured 1 --m_structured 2 --pattern nm --implicit --culinear --patch_size 1 --log_freq 1 --num_workers 16