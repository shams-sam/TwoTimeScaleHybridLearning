# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.007 --non-iid 1 --repeat 1 --rounds 50 --lut-int 0 --eut-int 2 --dry-run 0 &
# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.007 --non-iid 1 --repeat 1 --rounds 50 --lut-int 0 --eut-int 5 --dry-run 0 &
# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.007 --non-iid 1 --repeat 1 --rounds 50 --lut-int 0 --eut-int 10 --dry-run 0 &
# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.007 --non-iid 1 --repeat 1 --rounds 50 --lut-int 0 --eut-int 20 --dry-run 0 &
# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.007 --non-iid 1 --repeat 1 --rounds 50 --lut-int 0 --eut-int 30 --dry-run 0 &
# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.007 --non-iid 1 --repeat 1 --rounds 50 --lut-int 0 --eut-int 50 --dry-run 0


# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.007 --non-iid 1 --repeat 1 --rounds 50 --lut-int 1 --eut-int 10 --dry-run 0 &
# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.007 --non-iid 1 --repeat 1 --rounds 50 --lut-int 2 --eut-int 10 --dry-run 0 &
# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.007 --non-iid 1 --repeat 1 --rounds 50 --lut-int 5 --eut-int 10 --dry-run 0 &

# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.007 --non-iid 1 --repeat 1 --rounds 50 --lut-int 1 --eut-int 20 --dry-run 0 &
# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.007 --non-iid 1 --repeat 1 --rounds 50 --lut-int 2 --eut-int 20 --dry-run 0 &
# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.007 --non-iid 1 --repeat 1 --rounds 50 --lut-int 4 --eut-int 20 --dry-run 0 &
# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.007 --non-iid 1 --repeat 1 --rounds 50 --lut-int 5 --eut-int 20 --dry-run 0 &
# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.007 --non-iid 1 --repeat 1 --rounds 50 --lut-int 10 --eut-int 20 --dry-run 0 &

# decreasing lut increasing eut
# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.003 --non-iid 1 --repeat 1 --rounds 50 --sigma-mul 0 --lut-int 0 --eut-int 10 --dry-run 0 &
# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.003 --non-iid 1 --repeat 1 --rounds 0 --sigma-mul 0.01 --lut-int 4 --eut-int 20 --dry-run 0 &
# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.003 --non-iid 1 --repeat 1 --rounds 0 --sigma-mul 0.01 --lut-int 2 --eut-int 30 --dry-run 0 &
# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.003 --non-iid 1 --repeat 1 --rounds 0 --sigma-mul 0.01 --lut-int 1 --eut-int 40 --dry-run 0 &

# =========================================================
# increasing and decreasing the period of eut and comparing
# =========================================================
# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.003 --non-iid 1 --repeat 1 --rounds 50 --lut-int 0 --eut-int 40 --eut-gamma 0.8 --dry-run 0 &
# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.003 --non-iid 1 --repeat 1 --rounds 50 --lut-int 0 --eut-int 40 --eut-gamma 1.2 --dry-run 0 &
# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.003 --non-iid 1 --repeat 1 --rounds 50 --lut-int 0 --eut-int 40 --dry-run 0 &

# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.003 --non-iid 1 --repeat 1 --rounds 50 --lut-int 0 --eut-int 20 --eut-gamma 0.8 --dry-run 0 &
# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.003 --non-iid 1 --repeat 1 --rounds 50 --lut-int 0 --eut-int 20 --eut-gamma 1.2 --dry-run 0 &
# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.003 --non-iid 1 --repeat 1 --rounds 50 --lut-int 0 --eut-int 20 --dry-run 0 &

# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.003 --non-iid 1 --repeat 1 --rounds 50 --lut-int 0 --eut-int 10 --eut-gamma 0.8 --dry-run 0 &
# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.003 --non-iid 1 --repeat 1 --rounds 50 --lut-int 0 --eut-int 10 --eut-gamma 1.2 --dry-run 0 &
# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.003 --non-iid 1 --repeat 1 --rounds 50 --lut-int 0 --eut-int 10 --dry-run 0 &

# ====
# lut=eut
# ====

# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.003 --non-iid 1 --repeat 1 --rounds 50 --lut-int 40 --eut-int 40 --eut-gamma 0.8 --dry-run 0 &
# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.003 --non-iid 1 --repeat 1 --rounds 50 --lut-int 40 --eut-int 40 --eut-gamma 1.2 --dry-run 0 &
# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.003 --non-iid 1 --repeat 1 --rounds 50 --lut-int 40 --eut-int 40 --dry-run 0 &

# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.003 --non-iid 1 --repeat 1 --rounds 50 --lut-int 20 --eut-int 20 --eut-gamma 0.8 --dry-run 0 &
# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.003 --non-iid 1 --repeat 1 --rounds 50 --lut-int 20 --eut-int 20 --eut-gamma 1.2 --dry-run 0 &
# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.003 --non-iid 1 --repeat 1 --rounds 50 --lut-int 20 --eut-int 20 --dry-run 0 &

# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.003 --non-iid 1 --repeat 1 --rounds 50 --lut-int 20 --eut-int 10 --eut-gamma 0.8 --dry-run 0 &
# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.003 --non-iid 1 --repeat 1 --rounds 50 --lut-int 20 --eut-int 10 --eut-gamma 1.2 --dry-run 0 &
# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.003 --non-iid 1 --repeat 1 --rounds 50 --lut-int 20 --eut-int 10 --dry-run 0 &


# ====
# divergence factor
# ====
# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.003 --non-iid 1 --repeat 1 --rounds 5 --lut-int 10 --eut-int 10 --factor 2 --dry-run 0 &&
#     python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.003 --non-iid 1 --repeat 1 --rounds 5 --lut-int 20 --eut-int 20 --factor 2 --dry-run 0 &
# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.003 --non-iid 1 --repeat 1 --rounds 5 --lut-int 10 --eut-int 10 --factor 4 --dry-run 0 &&
#     python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.003 --non-iid 1 --repeat 1 --rounds 5 --lut-int 20 --eut-int 20 --factor 4 --dry-run 0 &
# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.003 --non-iid 1 --repeat 1 --rounds 5 --lut-int 10 --eut-int 10 --factor 8 --dry-run 0 &&
#     python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.003 --non-iid 1 --repeat 1 --rounds 5 --lut-int 20 --eut-int 20 --factor 8 --dry-run 0 &
# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.003 --non-iid 1 --repeat 1 --rounds 5 --lut-int 10 --eut-int 10 --factor 16 --dry-run 0 &&
#     python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.003 --non-iid 1 --repeat 1 --rounds 5 --lut-int 20 --eut-int 20 --factor 16 --dry-run 0 &
# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.003 --non-iid 1 --repeat 1 --rounds 5 --lut-int 10 --eut-int 10 --factor 32 --dry-run 0 &&
#     python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.003 --non-iid 1 --repeat 1 --rounds 5 --lut-int 20 --eut-int 20 --factor 32 --dry-run 0 &
# python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.003 --non-iid 1 --repeat 1 --rounds 5 --lut-int 10 --eut-int 10 --factor 64 --dry-run 0 &&
#     python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.003 --non-iid 1 --repeat 1 --rounds 5 --lut-int 20 --eut-int 20 --factor 64 --dry-run 0 &



python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 200 --lr 0.003 --non-iid 1 --repeat 1 --rounds 5 --lut-int 20 --eut-int 20 --factor 32 --dry-run 0 &
python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 200 --lr 0.003 --non-iid 1 --repeat 1 --rounds 5 --lut-int 10 --eut-int 20 --factor 32 --dry-run 0 &
python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 200 --lr 0.003 --non-iid 1 --repeat 1 --rounds 5 --lut-int 5 --eut-int 20 --factor 32 --dry-run 0 &
python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 200 --lr 0.003 --non-iid 1 --repeat 1 --rounds 5 --lut-int 4 --eut-int 20 --factor 32 --dry-run 0 &
python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 200 --lr 0.003 --non-iid 1 --repeat 1 --rounds 5 --lut-int 2 --eut-int 20 --factor 32 --dry-run 0 &
python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 200 --lr 0.003 --non-iid 1 --repeat 1 --rounds 5 --lut-int 1 --eut-int 20 --factor 32 --dry-run 0 &

python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 200 --lr 0.003 --non-iid 1 --repeat 1 --rounds 5 --lut-int 5 --eut-int 10 --factor 64 --dry-run 0 &
python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 200 --lr 0.003 --non-iid 1 --repeat 1 --rounds 5 --lut-int 2 --eut-int 10 --factor 64 --dry-run 0 &
python train_model.py --dataset mnist --clf svm --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 200 --lr 0.003 --non-iid 1 --repeat 1 --rounds 5 --lut-int 1 --eut-int 10 --factor 64 --dry-run 0 &
