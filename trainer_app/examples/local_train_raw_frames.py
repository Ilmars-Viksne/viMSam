"""
Local command example for viMSam Trainer.
"""

RAW_FRAME_TRAINING = """
python trainer_app/main.py \
  --images trainer_app/data/training/raw_frames \
  --masks trainer_app/data/training/masks \
  --out trainer_app/data/models/cell_model_v1 \
  --workflow raw_frames \
  --model vit_b \
  --epochs 50 \
  --batch-size 2 \
  --raw-width 1024 \
  --raw-height 1024 \
  --preprocessing-method fixed_16bit \
  --patch-shape 512,512 \
  --device auto
"""
