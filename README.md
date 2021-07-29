# Go3D
ModelNet 3D object classification

https://github.gatech.edu/tzhong9/Go3D.git

To setup GPU for Tensorflow:
https://www.tensorflow.org/install/gpu

```
conda create -n go3d python=3.8
conda activate go3d
pip install -r requirements.txt

python main.py --config ./configs/<config_files>

load_ext tensorboard
tensorboard --logdir <log_dir>
```
