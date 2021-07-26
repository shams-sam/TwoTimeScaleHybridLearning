debug(){
    python train_model.py --dataset mnist --clf fcn --paradigm fp --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 21 --lr 0.01 --non-iid 10 --repeat 1 --eut-range 10 --lut-intv 2 --rounds 50 --eut-seed 0 --accuracy 0.8 --patience 0 --channel 2 --channel 2 --dry-run 1
}

mnistfc(){
    # python train_model.py --dataset mnist --clf fcn --paradigm fpn --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 201 --lr 0.01 --non-iid 1 --repeat 1 --eut-seed 0 --accuracy 0.8 --patience 0 --dry-run 0 &
    # python train_model.py --dataset mnist --clf fcn --paradigm fpn --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 201 --lr 0.01 --non-iid 3 --repeat 1 --eut-seed 0 --accuracy 0.8 --patience 0 --dry-run 0 &
    python train_model.py --dataset mnist --clf fcn --paradigm fpn --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 201 --lr 0.01 --non-iid 10 --repeat 1 --eut-seed 0 --accuracy 0.8 --patience 0 --dry-run 0 &
} # fedprox centralized would be same as fl

mnistfp(){
    # python train_model.py --dataset mnist --clf fcn --paradigm fpn --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --eut-range 20 --eut-seed 0 --accuracy 0.8 --patience 0 --dry-run 0 &
    # python train_model.py --dataset mnist --clf fcn --paradigm fpn --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 101 --lr 0.01 --non-iid 3 --repeat 1 --eut-range 20 --eut-seed 0 --accuracy 0.8 --patience 0 --dry-run 0 &
    python train_model.py --dataset mnist --clf fcn --paradigm fpn --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --eut-range 20 --eut-seed 0 --accuracy 0.8 --patience 0 --dry-run 0  &
} # fedprox with delayed aggregation

mnist10(){
    python train_model.py --dataset mnist --clf fcn --paradigm fp --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --eut-range 20 --lut-intv 5 --rounds 1 --eut-seed 0 --accuracy 0.8 --patience 0 --channel 2 --dry-run 0 &
    python train_model.py --dataset mnist --clf fcn --paradigm fp --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --eut-range 20 --lut-intv 5 --rounds 5 --eut-seed 0 --accuracy 0.8 --patience 0 --channel 2 --dry-run 0 &
    python train_model.py --dataset mnist --clf fcn --paradigm fp --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --eut-range 20 --lut-intv 5 --rounds 10 --eut-seed 0 --accuracy 0.8 --patience 0 --channel 2 --dry-run 0 &
    python train_model.py --dataset mnist --clf fcn --paradigm fp --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --eut-range 20 --lut-intv 5 --rounds 25 --eut-seed 0 --accuracy 0.8 --patience 0 --channel 2 --dry-run 0 &
} #

mnist3(){
    python train_model.py --dataset mnist --clf fcn --paradigm fp --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 101 --lr 0.01 --non-iid 3 --repeat 1 --eut-range 20 --lut-intv 5 --rounds 1 --eut-seed 0 --accuracy 0.8 --patience 0 --channel 2 --dry-run 0 &
    python train_model.py --dataset mnist --clf fcn --paradigm fp --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 101 --lr 0.01 --non-iid 3 --repeat 1 --eut-range 20 --lut-intv 5 --rounds 5 --eut-seed 0 --accuracy 0.8 --patience 0 --channel 2 --dry-run 0 &
    python train_model.py --dataset mnist --clf fcn --paradigm fp --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 101 --lr 0.01 --non-iid 3 --repeat 1 --eut-range 20 --lut-intv 5 --rounds 10 --eut-seed 0 --accuracy 0.8 --patience 0 --channel 2 --dry-run 0 &
    python train_model.py --dataset mnist --clf fcn --paradigm fp --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 101 --lr 0.01 --non-iid 3 --repeat 1 --eut-range 20 --lut-intv 5 --rounds 25 --eut-seed 0 --accuracy 0.8 --patience 0 --channel 2 --dry-run 0 &
} #

mnist1(){
    python train_model.py --dataset mnist --clf fcn --paradigm fp --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --eut-range 20 --lut-intv 5 --rounds 1 --eut-seed 0  --accuracy 0.8 --patience 0 --channel 2 --dry-run 0 &
    python train_model.py --dataset mnist --clf fcn --paradigm fp --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --eut-range 20 --lut-intv 5 --rounds 5 --eut-seed 0 --accuracy 0.8 --patience 0 --channel 2 --dry-run 0 &
    python train_model.py --dataset mnist --clf fcn --paradigm fp --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --eut-range 20 --lut-intv 5 --rounds 10 --eut-seed 0 --accuracy 0.8 --patience 0 --channel 2 --dry-run 0 &
    python train_model.py --dataset mnist --clf fcn --paradigm fp --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --eut-range 20 --lut-intv 5 --rounds 25 --eut-seed 0 --accuracy 0.8 --patience 0 --channel 2 --dry-run 0 &
} #


fmnist10(){
    python train_model.py --dataset fmnist --clf fcn --paradigm fp --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --eut-range 20 --lut-intv 5 --rounds 1 --eut-seed 0 --accuracy 0.8 --patience 0 --channel 2 --dry-run 0 &
    python train_model.py --dataset fmnist --clf fcn --paradigm fp --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --eut-range 20 --lut-intv 5 --rounds 5 --eut-seed 0 --accuracy 0.8 --patience 0 --channel 2 --dry-run 0 &
    python train_model.py --dataset fmnist --clf fcn --paradigm fp --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --eut-range 20 --lut-intv 5 --rounds 10 --eut-seed 0 --accuracy 0.8 --patience 0 --channel 2 --dry-run 0 &
    python train_model.py --dataset fmnist --clf fcn --paradigm fp --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 101 --lr 0.01 --non-iid 10 --repeat 1 --eut-range 20 --lut-intv 5 --rounds 25 --eut-seed 0 --accuracy 0.8 --patience 0 --channel 2 --dry-run 0 &
} #

fmnist3(){
    python train_model.py --dataset fmnist --clf fcn --paradigm fp --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 101 --lr 0.01 --non-iid 3 --repeat 1 --eut-range 20 --lut-intv 5 --rounds 1 --eut-seed 0 --accuracy 0.8 --patience 0 --channel 2 --dry-run 0 &
    python train_model.py --dataset fmnist --clf fcn --paradigm fp --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 101 --lr 0.01 --non-iid 3 --repeat 1 --eut-range 20 --lut-intv 5 --rounds 5 --eut-seed 0 --accuracy 0.8 --patience 0 --channel 2 --dry-run 0 &
    python train_model.py --dataset fmnist --clf fcn --paradigm fp --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 101 --lr 0.01 --non-iid 3 --repeat 1 --eut-range 20 --lut-intv 5 --rounds 10 --eut-seed 0 --accuracy 0.8 --patience 0 --channel 2 --dry-run 0 &
    python train_model.py --dataset fmnist --clf fcn --paradigm fp --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 101 --lr 0.01 --non-iid 3 --repeat 1 --eut-range 20 --lut-intv 5 --rounds 25 --eut-seed 0  --accuracy 0.8 --patience 0 --channel 2 --dry-run 0 
} # 

fmnist1(){
    python train_model.py --dataset fmnist --clf fcn --paradigm fp --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --eut-range 20 --lut-intv 5 --rounds 1 --eut-seed 0  --accuracy 0.8 --patience 0 --channel 2 --dry-run 0 &
    python train_model.py --dataset fmnist --clf fcn --paradigm fp --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --eut-range 20 --lut-intv 5 --rounds 5 --eut-seed 0 --accuracy 0.8 --patience 0 --channel 2 --dry-run 0 &
    python train_model.py --dataset fmnist --clf fcn --paradigm fp --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --eut-range 20 --lut-intv 5 --rounds 10 --eut-seed 0 --accuracy 0.8 --patience 0 --channel 2 --dry-run 0 &
    python train_model.py --dataset fmnist --clf fcn --paradigm fp --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 101 --lr 0.01 --non-iid 1 --repeat 1 --eut-range 20 --lut-intv 5 --rounds 25 --eut-seed 0 --accuracy 0.8 --patience 0 --channel 2 --dry-run 0
} # 

$1
