#!/bin/bash

# Ctrl+C 입력 시 모든 백그라운드 작업 종료
trap "echo 'Stopping...'; pkill -P $$; exit" SIGINT

# 병렬 실행







# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/cifar10_c/source.yaml TEST.BATCH_SIZE 64 RNG_SEED 1  & sleep 1

#############################################################   CIFAR 10  - WRN  ##############################################################################################


# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/cifar10_c/tent.yaml TEST.BATCH_SIZE 16 RNG_SEED 1 WITH_BN False USE_BUFFERs True,False,False ALPHA_INIT 1e-4 & sleep 1
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/cifar10_c/eata.yaml TEST.BATCH_SIZE 16 RNG_SEED 1 WITH_BN False USE_BUFFERs True,False,False ALPHA_INIT 1e-4 & sleep 1


# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/cifar10_c/tent_buffer.yaml TEST.BATCH_SIZE 2 RNG_SEED 1 WITH_BN False USE_BUFFERs True,False,False ALPHA_INIT 1e-5 & sleep 1
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/cifar10_c/eata_buffer.yaml TEST.BATCH_SIZE 2 RNG_SEED 1 WITH_BN False USE_BUFFERs True,False,False ALPHA_INIT 1e-5 & sleep 1


# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/cifar10_c/tent_buffer.yaml TEST.BATCH_SIZE 4 RNG_SEED 1 WITH_BN False USE_BUFFERs True,False,False ALPHA_INIT 1e-5 & sleep 1
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/cifar10_c/eata_buffer.yaml TEST.BATCH_SIZE 4 RNG_SEED 1 WITH_BN False USE_BUFFERs True,False,False ALPHA_INIT 1e-5 & sleep 1


# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/cifar10_c/tent_buffer.yaml TEST.BATCH_SIZE 16 RNG_SEED 1 WITH_BN False USE_BUFFERs True,False,False ALPHA_INIT 1e-4 & sleep 1
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/cifar10_c/eata_buffer.yaml TEST.BATCH_SIZE 16 RNG_SEED 1 WITH_BN False USE_BUFFERs True,False,False ALPHA_INIT 1e-4 & sleep 1



#############################################################   CIFAR 10  - ResNeXT  ##############################################################################################


# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/cifar10_c_resnext/tent.yaml TEST.BATCH_SIZE 16 RNG_SEED 1 WITH_BN False USE_BUFFERs True,False,False ALPHA_INIT 1e-4 & sleep 1
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/cifar10_c_resnext/eata.yaml TEST.BATCH_SIZE 16 RNG_SEED 1 WITH_BN False USE_BUFFERs True,False,False ALPHA_INIT 1e-4 & sleep 1


# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/cifar10_c_resnext/tent_buffer.yaml TEST.BATCH_SIZE 2 RNG_SEED 1 WITH_BN False USE_BUFFERs True,False,False ALPHA_INIT 1e-5 & sleep 1
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/cifar10_c_resnext/eata_buffer.yaml TEST.BATCH_SIZE 2 RNG_SEED 1 WITH_BN False USE_BUFFERs True,False,False ALPHA_INIT 1e-5 & sleep 1

# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/cifar10_c_resnext/tent_buffer.yaml TEST.BATCH_SIZE 4 RNG_SEED 1 WITH_BN False USE_BUFFERs True,False,False ALPHA_INIT 1e-5 & sleep 1
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/cifar10_c_resnext/eata_buffer.yaml TEST.BATCH_SIZE 4 RNG_SEED 1 WITH_BN False USE_BUFFERs True,False,False ALPHA_INIT 1e-5 & sleep 1

# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/cifar10_c_resnext/tent_buffer.yaml TEST.BATCH_SIZE 16 RNG_SEED 1 WITH_BN False USE_BUFFERs True,False,False ALPHA_INIT 1e-4 & sleep 1
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/cifar10_c_resnext/eata_buffer.yaml TEST.BATCH_SIZE 16 RNG_SEED 1 WITH_BN False USE_BUFFERs True,False,False ALPHA_INIT 1e-4 & sleep 1


########################################################### CIFAR 100 - ResNeXt  ###################################################################################################

# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/cifar100_c/tent.yaml        TEST.BATCH_SIZE 4 RNG_SEED 1 WITH_BN False USE_BUFFERs True,False,False ALPHA_INIT 1e-5 & sleep 1
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/cifar100_c/eata.yaml        TEST.BATCH_SIZE 4 RNG_SEED 1 WITH_BN False USE_BUFFERs True,False,False ALPHA_INIT 1e-5 & sleep 1

# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/cifar100_c/tent_buffer.yaml TEST.BATCH_SIZE 2 RNG_SEED 1 WITH_BN False USE_BUFFERs True,False,False ALPHA_INIT 1e-5 & sleep 1
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/cifar100_c/eata_buffer.yaml TEST.BATCH_SIZE 2 RNG_SEED 1 WITH_BN False USE_BUFFERs True,False,False ALPHA_INIT 1e-5 & sleep 1

# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/cifar100_c/tent_buffer.yaml TEST.BATCH_SIZE 4 RNG_SEED 1 WITH_BN False USE_BUFFERs True,False,False ALPHA_INIT 1e-5 & sleep 1
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/cifar100_c/eata_buffer.yaml TEST.BATCH_SIZE 4 RNG_SEED 1 WITH_BN False USE_BUFFERs True,False,False ALPHA_INIT 1e-5 & sleep 1

# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/cifar100_c/tent_buffer.yaml TEST.BATCH_SIZE 16 RNG_SEED 1 WITH_BN False USE_BUFFERs True,True,False ALPHA_INIT 1e-4 & sleep 1
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/cifar100_c/eata_buffer.yaml TEST.BATCH_SIZE 16 RNG_SEED 1 WITH_BN False USE_BUFFERs True,True,False ALPHA_INIT 1e-4 & sleep 1



########################################################### CIFAR 100 - WRN 40  ###################################################################################################

# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/cifar100_c_wrn40/tent.yaml        TEST.BATCH_SIZE 2 RNG_SEED 1 WITH_BN False USE_BUFFERs True,False,False ALPHA_INIT 1e-5 & sleep 1
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/cifar100_c_wrn40/eata.yaml        TEST.BATCH_SIZE 2 RNG_SEED 1 WITH_BN False USE_BUFFERs True,False,False ALPHA_INIT 1e-5 & sleep 1

# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/cifar100_c_wrn40/tent_buffer.yaml TEST.BATCH_SIZE 2 RNG_SEED 1 WITH_BN False USE_BUFFERs True,False,False ALPHA_INIT 1e-5 & sleep 1
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/cifar100_c_wrn40/eata_buffer.yaml TEST.BATCH_SIZE 2 RNG_SEED 1 WITH_BN False USE_BUFFERs True,False,False ALPHA_INIT 1e-5 & sleep 1

# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/cifar100_c_wrn40/tent_buffer.yaml TEST.BATCH_SIZE 4 RNG_SEED 1 WITH_BN False USE_BUFFERs True,False,False ALPHA_INIT 1e-5 & sleep 1
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/cifar100_c_wrn40/eata_buffer.yaml TEST.BATCH_SIZE 4 RNG_SEED 1 WITH_BN False USE_BUFFERs True,False,False ALPHA_INIT 1e-5 & sleep 1

# CUDA_VISIBLE_DEVICES=2 python test_time.py --cfg cfgs/cifar100_c_wrn40/tent_buffer.yaml TEST.BATCH_SIZE 16 RNG_SEED 1 WITH_BN False USE_BUFFERs True,True,False ALPHA_INIT 1e-4 & sleep 1
# CUDA_VISIBLE_DEVICES=3 python test_time.py --cfg cfgs/cifar100_c_wrn40/eata_buffer.yaml TEST.BATCH_SIZE 16 RNG_SEED 1 WITH_BN False USE_BUFFERs True,True,False ALPHA_INIT 1e-4 & sleep 1





wait