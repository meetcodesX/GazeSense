---
dataset_info:
  features:
  - name: image
    dtype: image
  - name: gazes
    list:
    - name: head_bbox
      struct:
      - name: xmin
        dtype: float32
      - name: ymin
        dtype: float32
      - name: xmax
        dtype: float32
      - name: ymax
        dtype: float32
    - name: eye
      struct:
      - name: x
        dtype: float32
      - name: y
        dtype: float32
    - name: gaze
      struct:
      - name: x
        dtype: float32
      - name: y
        dtype: float32
    - name: body_bbox
      struct:
      - name: x
        dtype: float32
      - name: y
        dtype: float32
      - name: w
        dtype: float32
      - name: h
        dtype: float32
    - name: in_out
      dtype: int8
  splits:
  - name: test
    num_bytes: 241060099.472
    num_examples: 4782
  download_size: 238943038
  dataset_size: 241060099.472
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
---
