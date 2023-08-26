python scripts/evaluate_model.py DucoNet ./checkpoints/last_model/DucoNet256.pth \
--resize-strategy Fixed256 \
--gpu 0
#python scripts/evaluate_model.py DucoNet ./checkpoints/last_model/DucoNet1024.pth \
#--resize-strategy Fixed1024 \
#--gpu 1 \
#--datasets HAdobe5k1