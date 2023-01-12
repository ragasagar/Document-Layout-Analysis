#!/bin/bash
# ------------------------------------------------------------------
# [Author] Title
#          Description
# ------------------------------------------------------------------

rm -r publaynet
cp -r /home/venkat/targeted_learning_docbank/dataset/publaynet2000/ publaynet/
echo "copying dataset complete"
python talisman_subsetselection.py --output_path=talisman_test4_com --budget=50 --total_budget=1000 --strategy=com --lake_size=6896 --train_size=2000 --category=list --device=0
python informe.py