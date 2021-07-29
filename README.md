# Go3D (Authors: Gatech-CS7643-Group126)
ModelNet 3D object classification
```
https://github.gatech.edu/tzhong9/Go3D.git
```
3D objectives CAD data:
```
https://modelnet.cs.princeton.edu/
```
To setup GPU for Tensorflow:
```
https://www.tensorflow.org/install/gpu
```
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

Folder Structure:
```
Go3D
root:                 model execute script and tools       
  |--configs:         model configulations files
  |--data:            data rendering scripts
      |--datasets:    saved data 
      |--image:       saved images
  |--models:          model construction in keras
  |--outputs:         model outputs: parameter json file, model structure summary, model outputs
```
