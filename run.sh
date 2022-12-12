#!/bin/bash
# ------------------------------------------------------------------
# [Author] Title
#          Description
# ------------------------------------------------------------------

rm -r publaynet
cp -r /home/venkat/targeted_learning_docbank/olala_dataset/publaynet2000/ publaynet/
echo "copying dataset complete"
python talisman_subsetselection.py --output_path=talisman_test6 --budget=50 --total_budget=1000 --strategy=fl2mi --lake_size=6896 --train_size=2000 --category=list --device=0
python informe.py 
