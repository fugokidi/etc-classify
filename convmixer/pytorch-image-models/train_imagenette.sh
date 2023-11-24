# sh distributed_train.sh 2 /home/maung/imagenette2-320 --model convmixer_384_8 --amp -b 64 -j 8 --opt adamw --epochs 300 --sched onecycle --input-size 3 224 224 --lr 0.01 --cutmix 0.5 --mixup 0.5 --aa rand-m9-mstd0.5-inc1 --reprob 0.25 --remode pixel --num-classes 10 --opt-eps 1e-3 --clip-grad 1.0 --warmup-epochs 0 --weight-decay 1e-2

# sh distributed_train.sh 2 /home/maung/cifar10 --model tiny_convmixer --amp -b 64 -j 8 --opt adamw --epochs 300 --sched onecycle --input-size 3 32 32 --lr 0.01 --cutmix 0.5 --mixup 0.5 --aa rand-m9-mstd0.5-inc1 --num-classes 10 --opt-eps 1e-3 --clip-grad 1.0 --warmup-epochs 0 --scale 0.75 1.0 --weight-decay 1e-2

sh distributed_train.sh 1 /home/maung/imagenette2-320 --model convmixer_384_8 --amp -b 64 -j 8 --opt adamw --epochs 300 --sched onecycle --input-size 3 224 224 --lr 0.01 --cutmix 0.5 --mixup 0.5 --num-classes 10 --opt-eps 1e-3 --clip-grad 1.0 --warmup-epochs 0 --weight-decay 1e-2

# sh distributed_train.sh 1 /home/maung/cifar10_qf80_etc_old/ --model tiny_convmixer --amp -b 64 -j 8 --opt adamw --epochs 300 --sched onecycle --input-size 3 32 32 --lr 0.01  --cutmix 0.5 --mixup 0.5 --num-classes 10 --opt-eps 1e-3 --clip-grad 1.0 --warmup-epochs 0 --scale 0.75 1.0 --weight-decay 1e-2

# --no-aug
# --hflip 0

# sh distributed_train.sh 2 /home/maung/cifar10/ --model tiny_convmixer --amp -b 64 -j 8 --opt adamw --epochs 300 --sched onecycle --input-size 3 32 32 --lr 0.01 --no-aug --num-classes 10 --opt-eps 1e-3 --clip-grad 1.0 --warmup-epochs 0 --weight-decay 1e-2

# sh distributed_train.sh 1 /home/maung/cifar10_qf80_etc_old/ --model tiny_convmixer --amp -b 64 -j 8 --opt adamw --epochs 300 --sched onecycle --input-size 3 32 32 --lr 0.01 --no-aug --num-classes 10 --opt-eps 1e-3 --clip-grad 1.0 --warmup-epochs 0 --weight-decay 1e-2
