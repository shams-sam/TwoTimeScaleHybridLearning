debug(){
    python train_model.py --dataset mnist --clf fcn --paradigm fl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --eut-range 20 --eut-seed 0 --dry-run 1
    python train_model.py --dataset mnist --clf fcn --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 100 --lr 0.01 --non-iid 10 --repeat 1 --eut-range 20 --lut-intv 2 --rounds 5 --eut-seed 0 --dry-run 1
}

mnistfl(){
    python train_model.py --dataset mnist --clf fcn --paradigm fl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 200 --lr 0.01 --non-iid 1 --repeat 1 --eut-range 50 --eut-seed 0 --accuracy 0.8 --patience 0 --dry-run 0 &
    python train_model.py --dataset mnist --clf fcn --paradigm fl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 200 --lr 0.01 --non-iid 3 --repeat 1 --eut-range 50 --eut-seed 0 --accuracy 0.8 --patience 0 --dry-run 0 &
}


mnist1(){
    python train_model.py --dataset mnist --clf fcn --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 200 --lr 0.01 --non-iid 1 --repeat 1 --eut-range 60 --lut-intv 50 --rounds 1 --eut-seed 0 --accuracy 0.8 --patience 0 --dry-run 0 &
    python train_model.py --dataset mnist --clf fcn --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 200 --lr 0.01 --non-iid 1 --repeat 1 --eut-range 70 --lut-intv 50 --rounds 1 --eut-seed 0 --accuracy 0.8 --patience 0 --dry-run 0 &
    python train_model.py --dataset mnist --clf fcn --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 200 --lr 0.01 --non-iid 1 --repeat 1 --eut-range 90 --lut-intv 50 --rounds 1 --eut-seed 0 --accuracy 0.8 --patience 0 --dry-run 0 &
    python train_model.py --dataset mnist --clf fcn --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 200 --lr 0.01 --non-iid 1 --repeat 1 --eut-range 100 --lut-intv 50 --rounds 1 --eut-seed 0 --accuracy 0.8 --patience 0 --dry-run 0 &
} ##


fmnistfl(){
    python train_model.py --dataset fmnist --clf fcn --paradigm fl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 200 --lr 0.01 --non-iid 1 --repeat 1 --eut-range 50 --eut-seed 0  --accuracy 0.8 --patience 0 --dry-run 0 &
    python train_model.py --dataset fmnist --clf fcn --paradigm fl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 200 --lr 0.01 --non-iid 3 --repeat 1 --eut-range 50 --eut-seed 0  --accuracy 0.8 --patience 0 --dry-run 0
}

fmnist1(){
    python train_model.py --dataset fmnist --clf fcn --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 200 --lr 0.01 --non-iid 1 --repeat 1 --eut-range 60 --lut-intv 50 --rounds 1 --eut-seed 0  --accuracy 0.8 --patience 0 --dry-run 0 &
    python train_model.py --dataset fmnist --clf fcn --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 200 --lr 0.01 --non-iid 1 --repeat 1 --eut-range 70 --lut-intv 50 --rounds 1 --eut-seed 0  --accuracy 0.8 --patience 0 --dry-run 0 &
    python train_model.py --dataset fmnist --clf fcn --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 200 --lr 0.01 --non-iid 1 --repeat 1 --eut-range 90 --lut-intv 50 --rounds 1 --eut-seed 0  --accuracy 0.8 --patience 0 --dry-run 0 &
    python train_model.py --dataset fmnist --clf fcn --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 200 --lr 0.01 --non-iid 1 --repeat 1 --eut-range 100 --lut-intv 50 --rounds 1 --eut-seed 0  --accuracy 0.8 --patience 0 --dry-run 0
}

$1
