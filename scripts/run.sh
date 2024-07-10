#!/bin/bash

for epoch in 300
do
    for sample in 100
    do
        for reuploading in 3 5
        do  
            tr_sample=$((10*sample))
            nohup python -m run_classifier.py -e $epoch -te $sample -r $reuploading -tr $tr_sample > run_epoch_${epoch}_sample_${sample}_reuploading_${reuploading}.out &
            sleep 20
        done
    done
done