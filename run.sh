#!/bin/sh
path=Predataset/0/

for file in `ls Predataset/0`
do
    argv=$path$file
    echo $file
    python train.py $argv
done
