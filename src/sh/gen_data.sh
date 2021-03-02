mnist_25(){
    python gen_data.py --dataset mnist --num-workers 25 --non-iid 1 --repeat 1 --shuffle 1 --stratify 1 --uniform-data 1 --dry-run 0
    python gen_data.py --dataset mnist --num-workers 25 --non-iid 10 --repeat 1 --shuffle 1 --stratify 1 --uniform-data 1 --dry-run 0
}

fmnist_25(){
    python gen_data.py --dataset fmnist --num-workers 25 --non-iid 1 --repeat 1 --shuffle 1 --stratify 1 --uniform-data 1 --dry-run 0
    python gen_data.py --dataset fmnist --num-workers 25 --non-iid 10 --repeat 1 --shuffle 1 --stratify 1 --uniform-data 1 --dry-run 0
}

mnist_125(){
    python gen_data.py --dataset mnist --num-workers 125 --non-iid 1 --repeat 1 --shuffle 1 --stratify 1 --uniform-data 1 --dry-run 0
    python gen_data.py --dataset mnist --num-workers 125 --non-iid 3 --repeat 1 --shuffle 1 --stratify 1 --uniform-data 1 --dry-run 0
    python gen_data.py --dataset mnist --num-workers 125 --non-iid 10 --repeat 1 --shuffle 1 --stratify 1 --uniform-data 1 --dry-run 0
}

fmnist_125(){
    # python gen_data.py --dataset fmnist --num-workers 125 --non-iid 1 --repeat 1 --shuffle 1 --stratify 1 --uniform-data 1 --dry-run 0
    python gen_data.py --dataset fmnist --num-workers 125 --non-iid 3 --repeat 1 --shuffle 1 --stratify 1 --uniform-data 1 --dry-run 0
    # python gen_data.py --dataset fmnist --num-workers 125 --non-iid 10 --repeat 1 --shuffle 1 --stratify 1 --uniform-data 1 --dry-run 0
}


$1
