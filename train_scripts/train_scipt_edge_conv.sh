#!/bin/sh
#BSUB -J edgeconv-TRAIN #The name the job will get
#BSUB -q gpuv100 #The queue the job will be committed to, here the GPU enabled queue
#BSUB -gpu "num=1:mode=exclusive_process" #How the job will be run on the VM, here I request 1 GPU with exclusive access i.e. only my c #BSUB -n 1 How many CPU cores my job request
#BSUB -W 8:00 #The maximum runtime my job have note that the queuing might enable shorter jobs earlier due to scheduling.
#BSUB -R "span[hosts=1]" #How many nodes the job requests
#BSUB -R "rusage[mem=10GB]" #How much RAM the job should have access to
#BSUB -R "select[gpu32gb]" #For requesting the extra big GPU w. 32GB of VRAM
#BSUB -o logs/OUTPUT.%J #Log file
#BSUB -e logs/ERROR.%J #Error log file
echo "Starting:"

#cd ~/Documents/Thesis-paper/modelling/train_scripts
cd /zhome/2b/7/117471/Thesis-paper/modelling/train_scripts

#source ~/Documents/Thesis-paper/venv-paper/bin/activate
source /zhome/2b/7/117471/Thesis-paper/venv-paper/bin/activate

DATA=/zhome/2b/7/117471/Thesis-paper/data/hour1/biketown-portland-HOUR1-GRID5.pkl
#DATA=~/Documents/Thesis-paper/data/processed/hour1/biketown-portland-HOUR1-GRID5.pkl
MODEL=edgeconv
NUM_HISTORY=12
TRAIN_SIZE=0.9
BATCH_SIZE=24
EPOCHS=150
WEIGHT_DECAY=0.000000000001
LEARNING_RATE=0.0005
LR_FACTOR=0.1
LR_PATIENCE=25
OPTIMIZER=RMSprop
NODE_OUT_FEATURES=10
HIDDEN_SIZE=46
DROPOUT_P=0.2
SAVE_DIR=/zhome/2b/7/117471/Thesis-paper/models/test-models



python train_model.py --data $DATA --model $MODEL --num_history $NUM_HISTORY --train_size $TRAIN_SIZE \
--batch_size $BATCH_SIZE --epochs $EPOCHS --weight_decay $WEIGHT_DECAY --learning_rate $LEARNING_RATE \
--lr_factor $LR_FACTOR --lr_patience $LR_PATIENCE --optimizer $OPTIMIZER --hidden_size $HIDDEN_SIZE \
--node_out_feature $NODE_OUT_FEATURES --dropout $DROPOUT_P --save_dir $SAVE_DIR --gpu

