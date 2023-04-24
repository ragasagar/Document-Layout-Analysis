#!/bin/bash
# ------------------------------------------------------------------
# [Author] Title
#          Description
# ------------------------------------------------------------------

rm -r publaynet/
cp -r /home/venkat/targeted_learning_docbank/dataset/rebuttal/ publaynet/
# cp -r /home/venkat/targeted_learning_docbank/dataset/publaynet2000/ publaynet/
echo "copying dataset complete"
python subsetselection.py --output_path=rebuttal_5k/fl2mi --budget=200 --total_budget=2000 --strategy=fl2mi --lake_size=6896 --train_size=2000 --category=list --device=1
python informe.py