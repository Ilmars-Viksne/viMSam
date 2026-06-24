# viMSam Segmenter

Install the package in editable mode from the `segmenter_app` directory:

```bash
pip install -e .
```

Colab environments can continue to install the runtime stack with conda or mamba:

```bash
mamba install -y numpy=1.26.4 micro_sam pytorch=2.1.0 torchvision imageio imageio-ffmpeg tifffile python-dotenv pandas scikit-image ffmpeg -c pytorch -c conda-forge
pip install -e .
```

## CLI

Legacy and modern flags are both supported, including `--show_prompts`/`--show-prompts`, `--save_combined`/`--save-combined`, and `--tracking_method`/`--tracking-method`.

```bash
python main.py --input data/input/images/20260226213602.raw --out data/output/images/res.png --workflow raw_single --points "500,480" --show_prompts --save_combined
vimsam-segmenter --input data/input/videos/ --out data/output/videos/ --workflow raw_timeseries --points "500,480" --tracking-method pole --show-prompts --save-combined
vimsam-segmenter --input data/input/images/example.tif --out data/output/images/res.png --workflow single --points "150,100 200,400" --format csv
vimsam-segmenter --input data/input/videos/moving_cell.mp4 --out data/output/videos/video_box --workflow video --points "150,100" --tracking_method box --format json
```

Raw files default to headerless 16-bit unsigned 1024x1024 input. Use `--raw-width` and `--raw-height` for other dimensions.

## Python API

```python
from src import SegmenterApp, WorkflowConfig, PromptConfig

result = SegmenterApp().run(
    WorkflowConfig(
        workflow="raw_single",
        input_path="data/input/images/20260226213602.raw",
        output_path="data/output/images/res.png",
        prompts=PromptConfig(points=((500, 480),)),
        show_prompts=True,
        save_combined=True,
    )
)
```

## Migration Notes

The application logic lives under `segmenter_app/src`.
The command-line entry point `vimsam-segmenter` calls `src.cli:main` directly.
The legacy `vimsam_segmenter` compatibility package has been removed.
