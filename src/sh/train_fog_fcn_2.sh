python train_model.py --dataset mnist --clf fcn --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.01 --non-iid 10 --repeat 1 --rounds 50 --lut-int 2 --eut-int 1 --dry-run 0 &
python train_model.py --dataset mnist --clf fcn --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.01 --non-iid 10 --repeat 1 --rounds 50 --lut-int 2 --eut-int 5 --dry-run 0 &
python train_model.py --dataset mnist --clf fcn --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.01 --non-iid 10 --repeat 1 --rounds 50 --lut-int 2 --eut-int 10 --dry-run 0 &

python train_model.py --dataset mnist --clf fcn --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.01 --non-iid 1 --repeat 1 --rounds 50 --lut-int 2 --eut-int 1 --dry-run 0 &
python train_model.py --dataset mnist --clf fcn --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.01 --non-iid 1 --repeat 1 --rounds 50 --lut-int 2 --eut-int 5 --dry-run 0 &
python train_model.py --dataset mnist --clf fcn --paradigm fog --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.01 --non-iid 1 --repeat 1 --rounds 50 --lut-int 2 --eut-int 10 --dry-run 0

