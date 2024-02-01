#!/bin/bash

eps=0.0

echo python3 generate_data.py -e $eps
python3 generate_data.py -e $eps

mkdir results
for block in 512 1024 #16 32 64 128 256 512 1024
do
    echo "python3 myTrain.py -b $block | cut -d' ' -f5,9 | grep '^[0-9]' > results/obj_eps${eps}_block_${block}.txt"
    python3 myTrain.py -b $block | cut -d' ' -f5,9 | grep '^[0-9]' > results/obj_eps${eps}_block_${block}.txt

    echo "touch results/dis_diff_eps$eps_block${block}.txt"
    touch results/dis_diff_eps$eps_block${block}.txt
    
    for num in `seq 0 100 5001` 
    do	   
	echo "python3 TestConvergence.py  -n 1000 -M /tmp/my_model_$num | grep difference | cut -d' ' -f5 >> results/dis_diff_eps${eps}_block${block}.txt"
	python3 TestConvergence.py -n 1000 -M /tmp/my_model_$num | grep difference | cut -d' ' -f5 >> results/dis_diff_eps${eps}_block${block}.txt

    done  
done

