#!/bin/sh
#BSUB -J FINETUNE-MULTIPLE #The name the job will get
#BSUB -q gpuv100 #The queue the job will be committed to, here the GPU enabled queue
#BSUB -gpu "num=1:mode=exclusive_process" #How the job will be run on the VM, here I request 1 GPU with exclusive access i.e. only my c #BSUB -n 1 How many CPU cores my job request
#BSUB -W 24:00 #The maximum runtime my job have note that the queuing might enable shorter jobs earlier due to scheduling.
#BSUB -R "span[hosts=1]" #How many nodes the job requests
#BSUB -R "rusage[mem=12GB]" #How much RAM the job should have access to
#BSUB -R "select[gpu32gb]" #For requesting the extra big GPU w. 32GB of VRAM
#BSUB -o logs/OUTPUT.%J #Log file
#BSUB -e logs/ERROR.%J #Error log file
echo "Starting:"

cd ~/Thesis/metalearning
#cd /Users/theisferre/Documents/SPECIALE/Thesis/src/models

source ~/Thesis/venv-thesis/bin/activate

DATA=/zhome/2b/7/117471/Thesis/data/processed/metalearning/capitalbikeshare-tripdata-HOUR1-GRID10.pkl
MODEL_PATH=/zhome/2b/7/117471/Thesis/CASESTUDY/non-augmented/2021-11-03T14:53:15.962247
TRAIN_SIZE=0.9
BATCH_SIZE=20
EPOCHS=150
WEIGHT_DECAY=0.0000000001
LEARNING_RATE=0.0005
LR_PATIENCE=25
LR_FACTOR=0.1
OPTIMIZER=RMSprop
SAVE_DIR=/zhome/2b/7/117471/Thesis/CASESTUDY/metalearn_finetuned_multiple




python /zhome/2b/7/117471/Thesis/src/models/train_multiple_metalearn.py --data $DATA --model_path $MODEL_PATH --train_size $TRAIN_SIZE --batch_size $BATCH_SIZE --epochs $EPOCHS --weight_decay $WEIGHT_DECAY --learning_rate $LEARNING_RATE --lr_patience $LR_PATIENCE --lr_factor $LR_FACTOR --optimizer $OPTIMIZER --save_dir $SAVE_DIR --gpu

# TRAINED MODELS
# /zhome/2b/7/117471/Thesis/metalearning/2021-10-10T15:26:12.999216
# /zhome/2b/7/117471/Thesis/metalearning/2021-10-10T15:29:59.429279
# /zhome/2b/7/117471/Thesis/metalearning/2021-10-10T15:30:18.825955
# /zhome/2b/7/117471/Thesis/metalearning/2021-10-10T15:30:38.752281
# /zhome/2b/7/117471/Thesis/metalearning/2021-10-10T15:30:40.637652
# /zhome/2b/7/117471/Thesis/metalearning/2021-10-10T15:30:43.601055
# /zhome/2b/7/117471/Thesis/metalearning/2021-10-10T15:30:47.529250
# /zhome/2b/7/117471/Thesis/metalearning/2021-10-10T15:30:58.898052


