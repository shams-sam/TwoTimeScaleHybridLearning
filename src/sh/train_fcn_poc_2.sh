debug(){
    python train_model.py --dataset mnist --clf fcn --paradigm fl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 51 --lr 0.01 --non-iid 10 --repeat 1 --eut-range 20 --eut-seed 0 --dry-run 1
    python train_model.py --dataset mnist --clf fcn --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 100 --lr 0.01 --non-iid 10 --repeat 1 --eut-range 20 --lut-intv 2 --rounds 5 --eut-seed 0 --dry-run 1
}

# mnist
mnist10(){
    python train_model.py --dataset mnist --clf fcn --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 100 --lr 0.01 --non-iid 10 --repeat 1 --eut-range 25 --lut-intv 5 --rounds 1 --eut-seed 0 --accuracy 0.8 --patience 0 --dry-run 0 &
    python train_model.py --dataset mnist --clf fcn --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 100 --lr 0.01 --non-iid 10 --repeat 1 --eut-range 30 --lut-intv 5 --rounds 5 --eut-seed 0 --accuracy 0.8 --patience 0 --dry-run 0 &
    python train_model.py --dataset mnist --clf fcn --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 100 --lr 0.01 --non-iid 10 --repeat 1 --eut-range 40 --lut-intv 5 --rounds 10 --eut-seed 0 --accuracy 0.8 --patience 0 --dry-run 0 &
    python train_model.py --dataset mnist --clf fcn --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 100 --lr 0.01 --non-iid 10 --repeat 1 --eut-range 50 --lut-intv 5 --rounds 25 --eut-seed 0 --accuracy 0.8 --patience 0 --dry-run 0
}

mnist3(){
    python train_model.py --dataset mnist --clf fcn --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 100 --lr 0.01 --non-iid 3 --repeat 1 --eut-range 25 --lut-intv 5 --rounds 1 --eut-seed 0 --accuracy 0.8 --patience 0 --dry-run 0 &
    python train_model.py --dataset mnist --clf fcn --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 100 --lr 0.01 --non-iid 3 --repeat 1 --eut-range 30 --lut-intv 5 --rounds 5 --eut-seed 0 --accuracy 0.8 --patience 0 --dry-run 0 &
    python train_model.py --dataset mnist --clf fcn --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 100 --lr 0.01 --non-iid 3 --repeat 1 --eut-range 40 --lut-intv 5 --rounds 10 --eut-seed 0 --accuracy 0.8 --patience 0 --dry-run 0 &
    python train_model.py --dataset mnist --clf fcn --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 100 --lr 0.01 --non-iid 3 --repeat 1 --eut-range 50 --lut-intv 5 --rounds 25 --eut-seed 0 --accuracy 0.8 --patience 0 --dry-run 0
} # 

mnist1(){
    python train_model.py --dataset mnist --clf fcn --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 100 --lr 0.01 --non-iid 1 --repeat 1 --eut-range 25 --lut-intv 5 --rounds 1 --eut-seed 0 --accuracy 0.8 --patience 0 --dry-run 0 &
    python train_model.py --dataset mnist --clf fcn --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 100 --lr 0.01 --non-iid 1 --repeat 1 --eut-range 30 --lut-intv 5 --rounds 5 --eut-seed 0 --accuracy 0.8 --patience 0 --dry-run 0 &
    python train_model.py --dataset mnist --clf fcn --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 100 --lr 0.01 --non-iid 1 --repeat 1 --eut-range 40 --lut-intv 5 --rounds 10 --eut-seed 0 --accuracy 0.8 --patience 0  --dry-run 0 &
    python train_model.py --dataset mnist --clf fcn --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 100 --lr 0.01 --non-iid 1 --repeat 1 --eut-range 50 --lut-intv 5 --rounds 25 --eut-seed 0 --accuracy 0.8 --patience 0 --dry-run 0
}


fmnist10(){
    python train_model.py --dataset fmnist --clf fcn --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 100 --lr 0.01 --non-iid 10 --repeat 1 --eut-range 25 --lut-intv 5 --rounds 1 --eut-seed 0  --accuracy 0.8 --patience 0 --dry-run 0 &
    python train_model.py --dataset fmnist --clf fcn --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 100 --lr 0.01 --non-iid 10 --repeat 1 --eut-range 30 --lut-intv 5 --rounds 5 --eut-seed 0  --accuracy 0.8 --patience 0 --dry-run 0 &
    python train_model.py --dataset fmnist --clf fcn --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 100 --lr 0.01 --non-iid 10 --repeat 1 --eut-range 40 --lut-intv 5 --rounds 10 --eut-seed 0  --accuracy 0.8 --patience 0 --dry-run 0 &
    python train_model.py --dataset fmnist --clf fcn --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 100 --lr 0.01 --non-iid 10 --repeat 1 --eut-range 50 --lut-intv 5 --rounds 25 --eut-seed 0 --accuracy 0.8 --patience 0 --dry-run 0
}

fmnist3(){
    python train_model.py --dataset fmnist --clf fcn --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 100 --lr 0.01 --non-iid 3 --repeat 1 --eut-range 25 --lut-intv 5 --rounds 1 --eut-seed 0  --accuracy 0.8 --patience 0 --dry-run 0 &
    python train_model.py --dataset fmnist --clf fcn --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 100 --lr 0.01 --non-iid 3 --repeat 1 --eut-range 30 --lut-intv 5 --rounds 5 --eut-seed 0  --accuracy 0.8 --patience 0 --dry-run 0 &
    python train_model.py --dataset fmnist --clf fcn --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 100 --lr 0.01 --non-iid 3 --repeat 1 --eut-range 40 --lut-intv 5 --rounds 10 --eut-seed 0  --accuracy 0.8 --patience 0 --dry-run 0 &
    python train_model.py --dataset fmnist --clf fcn --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 100 --lr 0.01 --non-iid 3 --repeat 1 --eut-range 50 --lut-intv 5 --rounds 25 --eut-seed 0  --accuracy 0.8 --patience 0 --dry-run 0
}

fmnist1(){
    python train_model.py --dataset fmnist --clf fcn --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 100 --lr 0.01 --non-iid 1 --repeat 1 --eut-range 25 --lut-intv 5 --rounds 1 --eut-seed 0  --accuracy 0.8 --patience 0 --dry-run 0 &
    python train_model.py --dataset fmnist --clf fcn --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 100 --lr 0.01 --non-iid 1 --repeat 1 --eut-range 30 --lut-intv 5 --rounds 5 --eut-seed 0  --accuracy 0.8 --patience 0 --dry-run 0 &
    python train_model.py --dataset fmnist --clf fcn --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 100 --lr 0.01 --non-iid 1 --repeat 1 --eut-range 40 --lut-intv 5 --rounds 10 --eut-seed 0  --accuracy 0.8 --patience 0 --dry-run 0 &
    python train_model.py --dataset fmnist --clf fcn --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 100 --lr 0.01 --non-iid 1 --repeat 1 --eut-range 50 --lut-intv 5 --rounds 25 --eut-seed 0  --accuracy 0.8 --patience 0 --dry-run 0
} ##

$1
