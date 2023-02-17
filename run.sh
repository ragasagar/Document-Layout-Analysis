#!/bin/bash
# ------------------------------------------------------------------
# [Author] Title
#          Description
# ------------------------------------------------------------------

rm -r sanskrit_tada/
cp -r /home/venkat/targeted_learning_docbank/dataset/sanskrit_tada/ sanskrit_tada/
# cp -r /home/venkat/targeted_learning_docbank/dataset/publaynet2000/ publaynet/
echo "copying dataset complete"
python sanskrit_subsetselection.py --output_path=sanskrit_com1 --budget=30 --total_budget=300 --strategy=com --lake_size=6896 --train_size=2000 --category=Table --device=1
python informe.py