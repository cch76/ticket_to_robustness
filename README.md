# ticket-to-robustness

## Getting Started
### Requirements
```
* ubuntu 18.0.4, cuda11
* python 3.8.3
* torch >= 1.7.0
* torchvision >= 0.8.1 
```
### Datasets
* CIFAR10, MNIST
* CIFAR10-C(https://github.com/hendrycks/robustness), MNIST-C(https://github.com/google-research/mnist-c)

## Train
cifar.py, mnist.py
### Arguments
| Args 	| Type 	| Description 	| Default|
|---------|--------|----------------------------------------------------|:-----:|
| datasets | [str] | cifar10, mnist | cifar10 |
| workers | [int] | number of data loading workers | 4 |
| epochs 	| [int] 	| epochs | 200|
| train-batch 	| [int] 	| train batchsize| 128|
| test-batch 	| [int] 	| test batchsize| 128|
| lr 	| [int] 	| learning rate| 128|
| schedule 	| [int] 	| Decrease learning rate at these epochs| [120, 160]|
| gamma 	| [float] 	| LR is multiplied by gamma on schedule| 0.1|
| momentum 	| [float] 	| momentum| 0.9|
| weight-decay 	| [float] 	| weight decay| 5e-4|
| arch 	| [str]	| resnet20, resnet34, wrn34, conv2, conv4, conv6 | 	resnet34 |
| save_dir 	| [str] 	| save files path	|  results/ |
| data_dir 	| [str] 	| data path | ../data  |

```
# Examples 
python cifar.py --save_path ./output/cifar --arch resnet34 --datasets cifar10 --data_dir ./your_data_path
python mnist.py --save_path ./output/mnist --arch conv4 --datasets mnist --data_dir ./your_data_path
```

## iterative pruning
cifar_iterative.py, mnist_iterative.py

### Arguments
| Args 	| Type 	| Description 	| Default|
|---------|--------|----------------------------------------------------|:-----:|
| datasets | [str] | cifar10, mnist | cifar10 |
| workers | [int] | number of data loading workers | 4 |
| epochs 	| [int] 	| epochs | 200|
| train-batch 	| [int] 	| train batchsize| 128|
| test-batch 	| [int] 	| test batchsize| 128|
| lr 	| [int] 	| learning rate| 128|
| schedule 	| [int] 	| Decrease learning rate at these epochs| [120, 160]|
| gamma 	| [float] 	| LR is multiplied by gamma on schedule| 0.1|
| momentum 	| [float] 	| momentum| 0.9|
| weight-decay 	| [float] 	| weight decay| 5e-4|
| resume 	| [str]	| path to checkpoint | 	results/model_best.pth.tar |
| init 	| [str]	| path to initialize point | results/3ep.pth.tar |
| arch 	| [str]	| resnet20, resnet34, wrn34, conv2, conv4, conv6 | 	resnet34 |
| save_dir 	| [str] 	| save files path	|  results/ |
| data_dir 	| [str] 	| data path | ../data  |
| c_data_dir 	| [str] 	| data path | ../data/cifar-10-c  |


``` 
# Examples
python cifar_iterative.py --save_path ./output/cifar --arch resnet34 --datasets cifar10 --resume ./ your_resume_path --init your_init_path --data_dir ./your_data_path --c_data_dir ./your_c_data_path
python mnist_iterative.py --save_path ./output/mnist --arch conv4 --datasets mnist --resume ./ your_resume_path --init your_init_path --c_data_dir ./your_c_data_path
```


## with other method
### CBAM
``` 
# Examples
python train_cifar.py --arch resnet --datasets cifar10 --data_dir ./your_data_path --prefix cbam-resnet34-cifar --att-type CBAM
python iterative_pruning.py --arch resnet --datasets cifar10 --data_dir ./your_data_path --prefix cbam-resnet34-cifar --att-type CBAM --c_data_dir ./your_c_data_path
```

### AugMix
``` 
# Examples
python augmix_cifar.py --model resnet34 --datasets cifar10 --data_dir ./your_data_path --c_data_dir ./your_c_data_path --save ./your_save_dir
python iterative_pruning.py --model resnet34 --datasets cifar10 --data_dir ./your_data_path --c_data_dir ./your_c_data_path --save ./your_save_dir
```

### Contact for issues
- Choi ChanHee, cch@ds.seoultech.ac.kr
