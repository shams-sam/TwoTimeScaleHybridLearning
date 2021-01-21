# mnist
mnist_125(){
    python train_model.py --dataset mnist --clf fcn --paradigm fl --num-workers 125 --batch-size 128 --num-clusters 25 1 --epochs 100 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --dataset mnist --clf fcn --paradigm fl --num-workers 125 --batch-size 128 --num-clusters 25 1 --epochs 100 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
}

# fmnist
fmnist_125(){
    python train_model.py --dataset fmnist --clf fcn --paradigm fl --num-workers 125 --batch-size 128 --num-clusters 25 1 --epochs 100 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
    python train_model.py --dataset fmnist --clf fcn --paradigm fl --num-workers 125 --batch-size 128 --num-clusters 25 1 --epochs 100 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
}


$1
