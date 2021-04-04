debug(){
    python train_model.py --dataset mnist --clf fcn --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 151 --lr 0 --non-iid 1 --repeat 1 --eut-range 10 --eut-seed 0 --delta 10 --sigma 100 --zeta 10e-4 --beta 20.0 --mu 1.0 --phi 0 --xi 0.01 --tau-max 40 --e-frac 0.06 --d-frac 0.06 --accuracy 0.8 --patience 10 --dry-run 1
}

mnistc(){
    python train_model.py --dataset mnist --clf fcn --paradigm fl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 201 --lr 0.001 --non-iid 1 --repeat 1 --eut-seed 0 --accuracy 0.8 --patience 0 --dry-run 0 &
}

mnistfl(){
    python train_model.py --dataset mnist --clf fcn --paradigm fl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 351 --lr 0.001 --non-iid 1 --repeat 1 --eut-range 20 --eut-seed 0 --accuracy 0.8 --patience 0 --dry-run 0 &
}


mnist1_1(){
    python train_model.py --dataset mnist --clf fcn --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 101 --lr 0 --non-iid 1 --repeat 1 --eut-range 10 --eut-seed 0 --delta 10 --sigma 100 --zeta 10e-4 --beta 20.0 --mu 1.0 --phi 10.0 --xi 0.01 --tau-max 40 --e-frac 0.01 --d-frac 0.01 --accuracy 0.6 --patience 10 --dry-run 0 &
    python train_model.py --dataset mnist --clf fcn --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 101 --lr 0 --non-iid 1 --repeat 1 --eut-range 10 --eut-seed 0 --delta 10 --sigma 100 --zeta 10e-4 --beta 20.0 --mu 1.0 --phi 10.0 --xi 0.01 --tau-max 40 --e-frac 0.05 --d-frac 0.01 --accuracy 0.6 --patience 10 --dry-run 0 &
    python train_model.py --dataset mnist --clf fcn --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 101 --lr 0 --non-iid 1 --repeat 1 --eut-range 10 --eut-seed 0 --delta 10 --sigma 100 --zeta 10e-4 --beta 20.0 --mu 1.0 --phi 10.0 --xi 0.01 --tau-max 40 --e-frac 0.10 --d-frac 0.01 --accuracy 0.6 --patience 10 --dry-run 0 &
    python train_model.py --dataset mnist --clf fcn --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 101 --lr 0 --non-iid 1 --repeat 1 --eut-range 10 --eut-seed 0 --delta 10 --sigma 100 --zeta 10e-4 --beta 20.0 --mu 1.0 --phi 10.0 --xi 0.01 --tau-max 40 --e-frac 0.15 --d-frac 0.01 --accuracy 0.6 --patience 10 --dry-run 0 &
}

mnist1_2(){    
    python train_model.py --dataset mnist --clf fcn --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 101 --lr 0 --non-iid 1 --repeat 1 --eut-range 10 --eut-seed 0 --delta 10 --sigma 100 --zeta 10e-4 --beta 20.0 --mu 1.0 --phi 10.0 --xi 0.01 --tau-max 40 --e-frac 0.01 --d-frac 0.05 --accuracy 0.6 --patience 10 --dry-run 0 &
    python train_model.py --dataset mnist --clf fcn --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 101 --lr 0 --non-iid 1 --repeat 1 --eut-range 10 --eut-seed 0 --delta 10 --sigma 100 --zeta 10e-4 --beta 20.0 --mu 1.0 --phi 10.0 --xi 0.01 --tau-max 40 --e-frac 0.05 --d-frac 0.05 --accuracy 0.6 --patience 10 --dry-run 0 &
    python train_model.py --dataset mnist --clf fcn --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 101 --lr 0 --non-iid 1 --repeat 1 --eut-range 10 --eut-seed 0 --delta 10 --sigma 100 --zeta 10e-4 --beta 20.0 --mu 1.0 --phi 10.0 --xi 0.01 --tau-max 40 --e-frac 0.10 --d-frac 0.05 --accuracy 0.6 --patience 10 --dry-run 0 &
    python train_model.py --dataset mnist --clf fcn --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 101 --lr 0 --non-iid 1 --repeat 1 --eut-range 10 --eut-seed 0 --delta 10 --sigma 100 --zeta 10e-4 --beta 20.0 --mu 1.0 --phi 10.0 --xi 0.01 --tau-max 40 --e-frac 0.15 --d-frac 0.05 --accuracy 0.6 --patience 10 --dry-run 0 &
}

mnist1_3(){
    python train_model.py --dataset mnist --clf fcn --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 101 --lr 0 --non-iid 1 --repeat 1 --eut-range 10 --eut-seed 0 --delta 10 --sigma 100 --zeta 10e-4 --beta 20.0 --mu 1.0 --phi 10.0 --xi 0.01 --tau-max 40 --e-frac 0.01 --d-frac 0.10 --accuracy 0.6 --patience 10 --dry-run 0 &
    python train_model.py --dataset mnist --clf fcn --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 101 --lr 0 --non-iid 1 --repeat 1 --eut-range 10 --eut-seed 0 --delta 10 --sigma 100 --zeta 10e-4 --beta 20.0 --mu 1.0 --phi 10.0 --xi 0.01 --tau-max 40 --e-frac 0.05 --d-frac 0.10 --accuracy 0.6 --patience 10 --dry-run 0 &
    python train_model.py --dataset mnist --clf fcn --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 101 --lr 0 --non-iid 1 --repeat 1 --eut-range 10 --eut-seed 0 --delta 10 --sigma 100 --zeta 10e-4 --beta 20.0 --mu 1.0 --phi 10.0 --xi 0.01 --tau-max 40 --e-frac 0.10 --d-frac 0.10 --accuracy 0.6 --patience 10 --dry-run 0 &
    python train_model.py --dataset mnist --clf fcn --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 101 --lr 0 --non-iid 1 --repeat 1 --eut-range 10 --eut-seed 0 --delta 10 --sigma 100 --zeta 10e-4 --beta 20.0 --mu 1.0 --phi 10.0 --xi 0.01 --tau-max 40 --e-frac 0.15 --d-frac 0.10 --accuracy 0.6 --patience 10 --dry-run 0 &
}

mnist1_4(){
    python train_model.py --dataset mnist --clf fcn --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 101 --lr 0 --non-iid 1 --repeat 1 --eut-range 10 --eut-seed 0 --delta 10 --sigma 100 --zeta 10e-4 --beta 20.0 --mu 1.0 --phi 10.0 --xi 0.01 --tau-max 40 --e-frac 0.01 --d-frac 0.15 --accuracy 0.6 --patience 10 --dry-run 0 &
    python train_model.py --dataset mnist --clf fcn --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 101 --lr 0 --non-iid 1 --repeat 1 --eut-range 10 --eut-seed 0 --delta 10 --sigma 100 --zeta 10e-4 --beta 20.0 --mu 1.0 --phi 10.0 --xi 0.01 --tau-max 40 --e-frac 0.05 --d-frac 0.15 --accuracy 0.6 --patience 10 --dry-run 0 &
    python train_model.py --dataset mnist --clf fcn --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 101 --lr 0 --non-iid 1 --repeat 1 --eut-range 10 --eut-seed 0 --delta 10 --sigma 100 --zeta 10e-4 --beta 20.0 --mu 1.0 --phi 10.0 --xi 0.01 --tau-max 40 --e-frac 0.10 --d-frac 0.15 --accuracy 0.6 --patience 10 --dry-run 0 &
    python train_model.py --dataset mnist --clf fcn --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 101 --lr 0 --non-iid 1 --repeat 1 --eut-range 10 --eut-seed 0 --delta 10 --sigma 100 --zeta 10e-4 --beta 20.0 --mu 1.0 --phi 10.0 --xi 0.01 --tau-max 40 --e-frac 0.15 --d-frac 0.15 --accuracy 0.6 --patience 10 --dry-run 0 &
}

fmnistc(){
    python train_model.py --dataset fmnist --clf fcn --paradigm fl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 101 --lr 0.001 --non-iid 1 --repeat 1 --eut-seed 0 --accuracy 0.6 --patience 10 --dry-run 0 &
}

fmnistfl(){
    python train_model.py --dataset fmnist --clf fcn --paradigm fl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 351 --lr 0.001 --non-iid 1 --repeat 1 --eut-range 20 --eut-seed 0 --accuracy 0.6 --patience 10 --dry-run 0 &
}


fmnist1_1(){
    python train_model.py --dataset fmnist --clf fcn --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 101 --lr 0 --non-iid 1 --repeat 1 --eut-range 10 --eut-seed 0 --delta 10 --sigma 100 --zeta 10e-4 --beta 20.0 --mu 1.0 --phi 10.0 --xi 0.01 --tau-max 40 --e-frac 0.01 --d-frac 0.01 --accuracy 0.6 --patience 10 --dry-run 0 &
    python train_model.py --dataset fmnist --clf fcn --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 101 --lr 0 --non-iid 1 --repeat 1 --eut-range 10 --eut-seed 0 --delta 10 --sigma 100 --zeta 10e-4 --beta 20.0 --mu 1.0 --phi 10.0 --xi 0.01 --tau-max 40 --e-frac 0.05 --d-frac 0.01 --accuracy 0.6 --patience 10 --dry-run 0 &
    python train_model.py --dataset fmnist --clf fcn --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 101 --lr 0 --non-iid 1 --repeat 1 --eut-range 10 --eut-seed 0 --delta 10 --sigma 100 --zeta 10e-4 --beta 20.0 --mu 1.0 --phi 10.0 --xi 0.01 --tau-max 40 --e-frac 0.10 --d-frac 0.01 --accuracy 0.6 --patience 10 --dry-run 0 &
    python train_model.py --dataset fmnist --clf fcn --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 101 --lr 0 --non-iid 1 --repeat 1 --eut-range 10 --eut-seed 0 --delta 10 --sigma 100 --zeta 10e-4 --beta 20.0 --mu 1.0 --phi 10.0 --xi 0.01 --tau-max 40 --e-frac 0.15 --d-frac 0.01 --accuracy 0.6 --patience 10 --dry-run 0 &
}

fmnist1_2(){    
    python train_model.py --dataset fmnist --clf fcn --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 101 --lr 0 --non-iid 1 --repeat 1 --eut-range 10 --eut-seed 0 --delta 10 --sigma 100 --zeta 10e-4 --beta 20.0 --mu 1.0 --phi 10.0 --xi 0.01 --tau-max 40 --e-frac 0.01 --d-frac 0.05 --accuracy 0.6 --patience 10 --dry-run 0 &
    python train_model.py --dataset fmnist --clf fcn --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 101 --lr 0 --non-iid 1 --repeat 1 --eut-range 10 --eut-seed 0 --delta 10 --sigma 100 --zeta 10e-4 --beta 20.0 --mu 1.0 --phi 10.0 --xi 0.01 --tau-max 40 --e-frac 0.05 --d-frac 0.05 --accuracy 0.6 --patience 10 --dry-run 0 &
    python train_model.py --dataset fmnist --clf fcn --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 101 --lr 0 --non-iid 1 --repeat 1 --eut-range 10 --eut-seed 0 --delta 10 --sigma 100 --zeta 10e-4 --beta 20.0 --mu 1.0 --phi 10.0 --xi 0.01 --tau-max 40 --e-frac 0.10 --d-frac 0.05 --accuracy 0.6 --patience 10 --dry-run 0 &
    python train_model.py --dataset fmnist --clf fcn --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 101 --lr 0 --non-iid 1 --repeat 1 --eut-range 10 --eut-seed 0 --delta 10 --sigma 100 --zeta 10e-4 --beta 20.0 --mu 1.0 --phi 10.0 --xi 0.01 --tau-max 40 --e-frac 0.15 --d-frac 0.05 --accuracy 0.6 --patience 10 --dry-run 0 &
}

fmnist1_3(){
    python train_model.py --dataset fmnist --clf fcn --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 101 --lr 0 --non-iid 1 --repeat 1 --eut-range 10 --eut-seed 0 --delta 10 --sigma 100 --zeta 10e-4 --beta 20.0 --mu 1.0 --phi 10.0 --xi 0.01 --tau-max 40 --e-frac 0.01 --d-frac 0.10 --accuracy 0.6 --patience 10 --dry-run 0 &
    python train_model.py --dataset fmnist --clf fcn --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 101 --lr 0 --non-iid 1 --repeat 1 --eut-range 10 --eut-seed 0 --delta 10 --sigma 100 --zeta 10e-4 --beta 20.0 --mu 1.0 --phi 10.0 --xi 0.01 --tau-max 40 --e-frac 0.05 --d-frac 0.10 --accuracy 0.6 --patience 10 --dry-run 0 &
    python train_model.py --dataset fmnist --clf fcn --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 101 --lr 0 --non-iid 1 --repeat 1 --eut-range 10 --eut-seed 0 --delta 10 --sigma 100 --zeta 10e-4 --beta 20.0 --mu 1.0 --phi 10.0 --xi 0.01 --tau-max 40 --e-frac 0.10 --d-frac 0.10 --accuracy 0.6 --patience 10 --dry-run 0 &
    python train_model.py --dataset fmnist --clf fcn --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 101 --lr 0 --non-iid 1 --repeat 1 --eut-range 10 --eut-seed 0 --delta 10 --sigma 100 --zeta 10e-4 --beta 20.0 --mu 1.0 --phi 10.0 --xi 0.01 --tau-max 40 --e-frac 0.15 --d-frac 0.10 --accuracy 0.6 --patience 10 --dry-run 0 &
}

fmnist1_4(){
    python train_model.py --dataset fmnist --clf fcn --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 101 --lr 0 --non-iid 1 --repeat 1 --eut-range 10 --eut-seed 0 --delta 10 --sigma 100 --zeta 10e-4 --beta 20.0 --mu 1.0 --phi 10.0 --xi 0.01 --tau-max 40 --e-frac 0.01 --d-frac 0.15 --accuracy 0.6 --patience 10 --dry-run 0 &
    python train_model.py --dataset fmnist --clf fcn --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 101 --lr 0 --non-iid 1 --repeat 1 --eut-range 10 --eut-seed 0 --delta 10 --sigma 100 --zeta 10e-4 --beta 20.0 --mu 1.0 --phi 10.0 --xi 0.01 --tau-max 40 --e-frac 0.05 --d-frac 0.15 --accuracy 0.6 --patience 10 --dry-run 0 &
    python train_model.py --dataset fmnist --clf fcn --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 101 --lr 0 --non-iid 1 --repeat 1 --eut-range 10 --eut-seed 0 --delta 10 --sigma 100 --zeta 10e-4 --beta 20.0 --mu 1.0 --phi 10.0 --xi 0.01 --tau-max 40 --e-frac 0.10 --d-frac 0.15 --accuracy 0.6 --patience 10 --dry-run 0 &
    python train_model.py --dataset fmnist --clf fcn --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 101 --lr 0 --non-iid 1 --repeat 1 --eut-range 10 --eut-seed 0 --delta 10 --sigma 100 --zeta 10e-4 --beta 20.0 --mu 1.0 --phi 10.0 --xi 0.01 --tau-max 40 --e-frac 0.15 --d-frac 0.15 --accuracy 0.6 --patience 10 --dry-run 0 &
}


$1
