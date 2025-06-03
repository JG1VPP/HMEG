#!/usr/bin/env bash

if [ $# -lt 1 ] 
then
    echo "Usage: bash $0 GPUS"
    exit
fi

BIN=${BIN:-python3}
GPUS=$1

PORT=${PORT:-29500}
SCRIPT=$(dirname $0)/train.py

if [ ${GPUS} == 1 ]; then
    $BIN $SCRIPT
else
    $BIN -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT $SCRIPT --launcher pytorch
fi
