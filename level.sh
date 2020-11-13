# PyNET_smaller
python3 train_level.py --outtype PyNET_smaller --channel 16 --model-path ./model/PyNET_smaller-16 --batch-size 50 --nb-epochs 500 --level 5
python3 train_level.py --outtype PyNET_smaller --channel 16 --model-path ./model/PyNET_smaller-16 --batch-size 50 --nb-epochs 500 --level 4
python3 train_level.py --outtype PyNET_smaller --channel 16 --model-path ./model/PyNET_smaller-16 --batch-size 50 --nb-epochs 1000 --level 3
python3 train_level.py --outtype PyNET_smaller --channel 16 --model-path ./model/PyNET_smaller-16 --batch-size 50 --nb-epochs 1000 --level 2
python3 train_level.py --outtype PyNET_smaller --channel 16 --model-path ./model/PyNET_smaller-16 --batch-size 16 --nb-epochs 1500 --level 1
python3 train_level.py --outtype PyNET_smaller --channel 16 --model-path ./model/PyNET_smaller-16 --batch-size 16 --nb-epochs 1500 --level 0

# SE_ResNet
python3 train_level.py --outtype SE_ResNet --channel 32 --model-path ./model/SE_ResNet-32 --batch-size 50 --nb-epochs 500 --level 5
python3 train_level.py --outtype SE_ResNet --channel 32 --model-path ./model/SE_ResNet-32 --batch-size 50 --nb-epochs 500 --level 4
python3 train_level.py --outtype SE_ResNet --channel 32 --model-path ./model/SE_ResNet-32 --batch-size 50 --nb-epochs 1000 --level 3
python3 train_level.py --outtype SE_ResNet --channel 32 --model-path ./model/SE_ResNet-32 --batch-size 50 --nb-epochs 1000 --level 2
python3 train_level.py --outtype SE_ResNet --channel 32 --model-path ./model/SE_ResNet-32 --batch-size 16 --nb-epochs 1500 --level 1
python3 train_level.py --outtype SE_ResNet --channel 32 --model-path ./model/SE_ResNet-32 --batch-size 16 --nb-epochs 1500 --level 0