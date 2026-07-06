### N8 AOF Automation
- All you need to do in order to run this:
1. Download `yolo11n.pt` on Release for window detection
2. Download `yolo11n-cls.pt` on Release for card recognition

### Window Detection
- Auto detecting location of Natural8 window on a screen
- Download `yolo11n.pt` on Release, or
- if you wanna train one yourself:
```sh
pip install -r requirements.txt
cd window/
python window_data_generator.py

# this will download yolo11n model for the first time;
yolo classify train data=window_data.yaml model=yolo11n_custom.pt epochs=50 imgsz=640 

```