# python train_model.py --dataset fmnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.0035 --non-iid 1 --repeat 1 --rounds 50 --lut-int 0 --eut-int 2 --dry-run 0 &
# python train_model.py --dataset fmnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.0035 --non-iid 1 --repeat 1 --rounds 50 --lut-int 0 --eut-int 5 --dry-run 0 &
# python train_model.py --dataset fmnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.0035 --non-iid 1 --repeat 1 --rounds 50 --lut-int 0 --eut-int 10 --dry-run 0 &
# python train_model.py --dataset fmnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.0035 --non-iid 1 --repeat 1 --rounds 50 --lut-int 0 --eut-int 20 --dry-run 0 &
# python train_model.py --dataset fmnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.0035 --non-iid 1 --repeat 1 --rounds 50 --lut-int 0 --eut-int 30 --dry-run 0 &
# python train_model.py --dataset fmnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.0035 --non-iid 1 --repeat 1 --rounds 50 --lut-int 0 --eut-int 50 --dry-run 0 

# increasing eut decreasing lut similar performance
# python train_model.py --dataset fmnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.002 --non-iid 1 --repeat 1 --rounds 50 --sigma-mul 0 --lut-int 0 --eut-int 10 --dry-run 0 &
# python train_model.py --dataset fmnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.002 --non-iid 1 --repeat 1 --rounds 0 --sigma-mul 0.01 --lut-int 4 --eut-int 20 --dry-run 0 &

# python train_model.py --dataset fmnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.002 --non-iid 1 --repeat 1 --rounds 0 --sigma-mul 0.01 --lut-int 2 --eut-int 30 --dry-run 0 &
# python train_model.py --dataset fmnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.002 --non-iid 1 --repeat 1 --rounds 0 --sigma-mul 0.01 --lut-int 1 --eut-int 40 --dry-run 0 &

# ========================================================================
# varying intervals of eut within a single run (increasing and decreasing)
# ========================================================================
# python train_model.py --dataset fmnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.002 --non-iid 1 --repeat 1 --rounds 50 --lut-int 0 --eut-int 40 --eut-gamma 0.8 --dry-run 0 &
# python train_model.py --dataset fmnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.002 --non-iid 1 --repeat 1 --rounds 50 --lut-int 0 --eut-int 40 --eut-gamma 1.2 --dry-run 0 &
# python train_model.py --dataset fmnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.002 --non-iid 1 --repeat 1 --rounds 50 --lut-int 0 --eut-int 40 --dry-run 0 &

# python train_model.py --dataset fmnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.002 --non-iid 1 --repeat 1 --rounds 50 --lut-int 0 --eut-int 20 --eut-gamma 0.8 --dry-run 0 &
# python train_model.py --dataset fmnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.002 --non-iid 1 --repeat 1 --rounds 50 --lut-int 0 --eut-int 20 --eut-gamma 1.2 --dry-run 0 &
# python train_model.py --dataset fmnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.002 --non-iid 1 --repeat 1 --rounds 50 --lut-int 0 --eut-int 20 --dry-run 0 &

python train_model.py --dataset fmnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.002 --non-iid 1 --repeat 1 --rounds 50 --lut-int 0 --eut-int 10 --eut-gamma 0.8 --dry-run 0 &
python train_model.py --dataset fmnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.002 --non-iid 1 --repeat 1 --rounds 50 --lut-int 0 --eut-int 10 --eut-gamma 1.2 --dry-run 0 &
python train_model.py --dataset fmnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.002 --non-iid 1 --repeat 1 --rounds 50 --lut-int 0 --eut-int 10 --dry-run 0 &
