#!/bin/bash
# ------------------------------------------------------------------
# [Author] Title
#          Description
# ------------------------------------------------------------------

rm -r publaynet0/
cp -r /home/venkat/targeted_learning_docbank/dataset/rebuttal/ publaynet0/
# cp -r /home/venkat/targeted_learning_docbank/dataset/publaynet2000/ publaynet/
echo "copying dataset complete"
python subsetselection.py --output_path=rebuttal_5k/gcmi --budget=200 --total_budget=2000 --strategy=gcmi --lake_size=6896 --train_size=2000 --category=list --device=0
python informe.py