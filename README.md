# AOF built on Python
```python
python3 -m venv .env
source .env/bin/active
python3 -r requirements.txt
python3 multitable.py
```

```mermaid
flowchart LR
    A[MSS] --> B[YOLO]
    B --> C[Multitable Yolo]
    C --> D[Yolo card identification]
    D --> E[Strategy csv look-up]
    E --> F{Fold / All-in}
    F --> G[PyAutoGUI]
    C --> H[live_roi_*.png]
    H --> I[viewer.html]