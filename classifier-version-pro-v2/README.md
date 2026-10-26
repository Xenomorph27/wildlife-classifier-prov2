# classifier-version-pro-v2

YOLOv8x detection + EfficientNet-B4 classification across 52 wildlife species.
See the [project README](../README.md) for the full overview.

```bash
pip install -r ../requirements.txt
python predict.py                                   # bundled test.jpg
python predict.py my_photo.jpg --save out/ --min-conf 50
```
