"""Colab-oriented command examples for viMSam.

Install dependencies with conda/mamba, then run:

    pip install -e /content/segmenter_app
"""

RAW_SINGLE = """
python /content/segmenter_app/main.py \
    --input "/content/segmenter_app/data/input/images/20260226213602.raw" \
    --out "/content/segmenter_app/data/output/images/res.png" \
    --workflow "raw_single" \
    --points "500,480" \
    --show_prompts \
    --save_combined
"""

RAW_TIMESERIES = """
python /content/segmenter_app/main.py \
    --input "/content/segmenter_app/data/input/videos/" \
    --out "/content/segmenter_app/data/output/videos/" \
    --workflow "raw_timeseries" \
    --points "500,480" \
    --tracking_method pole \
    --show_prompts \
    --save_combined
"""
