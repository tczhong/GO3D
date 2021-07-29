# Go3D
ModelNet 3D object classification

https://github.gatech.edu/tzhong9/Go3D.git

To setup GPU for Tensorflow:
https://www.tensorflow.org/install/gpu

Linux GPU setup:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64
```

Create virtual environment and install necessary package:
```
conda create -n go3d python=3.8
conda activate go3d
pip install -r requirements.txt
```

Run the model:
```
python main.py --config ./configs/<config_files>
```

Evaluate the model performance:
```
tensorboard --logdir <log_dir>
```
