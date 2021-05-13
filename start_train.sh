nohup docker run --rm --cpuset-cpus 0 --gpus all -v $(pwd):/app voice-change python train.py &
