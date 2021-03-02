debug(){
    python train_model.py --dataset mnist --clf svm --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 51 --lr 0 --non-iid 10 --repeat 1 --eut-range 2 8 --eut-seed 0 --delta 10 --zeta 10e-4 --beta 20 --mu 1 --phi 10.0 --tau-max 25 --e-frac 0.04 --d-frac 0.01 --dry-run 1
}
mnist(){
    python train_model.py --dataset mnist --clf svm --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 51 --lr 0 --non-iid 1 --repeat 1 --eut-range 10 20 --eut-seed 0 --delta 10 --zeta 10e-4 --beta 20 --mu 1 --phi 0.001 --dry-run 0 &
    python train_model.py --dataset mnist --clf svm --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 51 --lr 0 --non-iid 1 --repeat 1 --eut-range 10 20 --eut-seed 0 --delta 10 --zeta 10e-4 --beta 20 --mu 1 --phi 0.1 --dry-run 0 &
    python train_model.py --dataset mnist --clf svm --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 51 --lr 0 --non-iid 1 --repeat 1 --eut-range 10 20 --eut-seed 0 --delta 10 --zeta 10e-4 --beta 20 --mu 1 --phi 1.0 --dry-run 0 &
    python train_model.py --dataset mnist --clf svm --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 51 --lr 0 --non-iid 1 --repeat 1 --eut-range 10 20 --eut-seed 0 --delta 10 --zeta 10e-4 --beta 20 --mu 1 --phi 10.0 --dry-run 0 &

    python train_model.py --dataset mnist --clf svm --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 51 --lr 0 --non-iid 10 --repeat 1 --eut-range 10 20 --eut-seed 0 --delta 10 --zeta 10e-4 --beta 20 --mu 1 --phi 0.001 --dry-run 0 &
    python train_model.py --dataset mnist --clf svm --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 51 --lr 0 --non-iid 10 --repeat 1 --eut-range 10 20 --eut-seed 0 --delta 10 --zeta 10e-4 --beta 20 --mu 1 --phi 0.1 --dry-run 0 &
    python train_model.py --dataset mnist --clf svm --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 51 --lr 0 --non-iid 10 --repeat 1 --eut-range 10 20 --eut-seed 0 --delta 10 --zeta 10e-4 --beta 20 --mu 1 --phi 1.0 --dry-run 0 &
    python train_model.py --dataset mnist --clf svm --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 51 --lr 0 --non-iid 10 --repeat 1 --eut-range 10 20 --eut-seed 0 --delta 10 --zeta 10e-4 --beta 20 --mu 1 --phi 10.0 --dry-run 0 &
}

fmnist(){
    python train_model.py --dataset fmnist --clf svm --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 51 --lr 0 --non-iid 1 --repeat 1 --eut-range 10 20 --eut-seed 0 --delta 10 --zeta 10e-4 --beta 20 --mu 1 --phi 0.001 --dry-run 0 &
    python train_model.py --dataset fmnist --clf svm --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 51 --lr 0 --non-iid 1 --repeat 1 --eut-range 10 20 --eut-seed 0 --delta 10 --zeta 10e-4 --beta 20 --mu 1 --phi 0.1 --dry-run 0 &
    python train_model.py --dataset fmnist --clf svm --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 51 --lr 0 --non-iid 1 --repeat 1 --eut-range 10 20 --eut-seed 0 --delta 10 --zeta 10e-4 --beta 20 --mu 1 --phi 1.0 --dry-run 0 &
    python train_model.py --dataset fmnist --clf svm --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 51 --lr 0 --non-iid 1 --repeat 1 --eut-range 10 20 --eut-seed 0 --delta 10 --zeta 10e-4 --beta 20 --mu 1 --phi 10.0 --dry-run 0 &

    python train_model.py --dataset fmnist --clf svm --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 51 --lr 0 --non-iid 10 --repeat 1 --eut-range 10 20 --eut-seed 0 --delta 10 --zeta 10e-4 --beta 20 --mu 1 --phi 0.001 --dry-run 0 &
    python train_model.py --dataset fmnist --clf svm --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 51 --lr 0 --non-iid 10 --repeat 1 --eut-range 10 20 --eut-seed 0 --delta 10 --zeta 10e-4 --beta 20 --mu 1 --phi 0.1 --dry-run 0 &
    python train_model.py --dataset fmnist --clf svm --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 51 --lr 0 --non-iid 10 --repeat 1 --eut-range 10 20 --eut-seed 0 --delta 10 --zeta 10e-4 --beta 20 --mu 1 --phi 1.0 --dry-run 0 &
    python train_model.py --dataset fmnist --clf svm --paradigm hl --num-workers 125 --num-clusters 25 1 --batch-size 16 --epochs 51 --lr 0 --non-iid 10 --repeat 1 --eut-range 10 20 --eut-seed 0 --delta 10 --zeta 10e-4 --beta 20 --mu 1 --phi 10.0 --dry-run 0 &
}


$1
