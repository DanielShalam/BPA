# Datasets

Follow the instructions to prepare the datasets.

## Miniimagenet:

```
./miniimagenet/
└── ./miniimagenet/split/                       # split files
    ├── ./miniimagenet/split/train.csv  
    ├── ./miniimagenet/split/val.csv  
    └── ./miniimagenet/split/test.csv  
└── /miniimagenet/images/                       # all images
    └── ./miniimagenet/images/<image_name>.jpg
   
```

Download the dataset from [here](https://cseweb.ucsd.edu/~weijian/static/datasets/mini-ImageNet/MiniImagenet.tar.gz).

Download the train/val/test split from this [repo](https://github.com/twitter-research/meta-learning-lstm/).

## CIFAR-FS:

```
./cifar_fs/
└── ./cifar_fs/train/                           # train images
    └── ./cifar_fs/train/<class_name>/<class_image>.jpg 
└── ./cifar_fs/val/                             # val images
    └── ./cifar_fs/val/<class_name>/<class_image>.jpg 
└── ./cifar_fs/test/                            # test images
    └── ./cifar_fs/test/<class_name>/<class_image>.jpg 
```

Run the following command to automatically download the dataset:

```
python get_cifar_fs.py
```

Modify "datapath" inside the script to choose different location.