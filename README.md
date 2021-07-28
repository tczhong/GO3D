# Go3D
ModelNet 3D object classification

https://github.gatech.edu/tzhong9/Go3D.git

```
conda create -n go3d python=3.8
conda activate go3d
pip install -r requirements.txt

cd data
python data.py
cd ..

python main.py --config ./configs/<config_files>

load_ext tensorboard
tensorboard --logdir <log_dir>
```
