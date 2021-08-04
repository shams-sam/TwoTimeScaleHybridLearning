debug(){
    python train_model.py --dataset mnist --clf svm --paradigm fl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 51 --lr 0.001 --non-iid 10 --repeat 1 --eut-range 20 --eut-seed 0 --dry-run 1
    python train_model.py --dataset mnist --clf svm --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 100 --lr 0.001 --non-iid 10 --repeat 1 --eut-range 20 --lut-intv 2 --rounds 5 --eut-seed 0 --dry-run 1
}

mnist1(){
    python train_model.py --dataset mnist --clf svm --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 200 --lr 0.001 --non-iid 1 --repeat 1 --eut-range 60 --lut-intv 50 --rounds 1 --eut-seed 0  --accuracy 0.8 --patience 0 --channel 2 --dry-run 0 &
    python train_model.py --dataset mnist --clf svm --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 200 --lr 0.001 --non-iid 1 --repeat 1 --eut-range 70 --lut-intv 50 --rounds 1 --eut-seed 0  --accuracy 0.8 --patience 0 --channel 2 --dry-run 0 &
    python train_model.py --dataset mnist --clf svm --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 200 --lr 0.001 --non-iid 1 --repeat 1 --eut-range 90 --lut-intv 50 --rounds 1 --eut-seed 0  --accuracy 0.8 --patience 0 --channel 2 --dry-run 0 &
    python train_model.py --dataset mnist --clf svm --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 200 --lr 0.001 --non-iid 1 --repeat 1 --eut-range 100 --lut-intv 50 --rounds 1 --eut-seed 0  --accuracy 0.8 --patience 0 --channel 2 --dry-run 0 &
}


fmnist1(){
    python train_model.py --dataset fmnist --clf svm --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 200 --lr 0.001 --non-iid 1 --repeat 1 --eut-range 60 --lut-intv 50 --rounds 1 --eut-seed 0  --accuracy 0.8 --patience 0 --channel 2 --dry-run 0 &
    python train_model.py --dataset fmnist --clf svm --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 200 --lr 0.001 --non-iid 1 --repeat 1 --eut-range 70 --lut-intv 50 --rounds 1 --eut-seed 0  --accuracy 0.8 --patience 0 --channel 2 --dry-run 0 &
    python train_model.py --dataset fmnist --clf svm --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 200 --lr 0.001 --non-iid 1 --repeat 1 --eut-range 90 --lut-intv 50 --rounds 1 --eut-seed 0  --accuracy 0.8 --patience 0 --channel 2 --dry-run 0 &
    python train_model.py --dataset fmnist --clf svm --paradigm hl --num-workers 125 --batch-size 16 --num-clusters 25 1 --epochs 200 --lr 0.001 --non-iid 1 --repeat 1 --eut-range 100 --lut-intv 50 --rounds 1 --eut-seed 0  --accuracy 0.8 --patience 0 --channel 2 --dry-run 0 &
}

$1
