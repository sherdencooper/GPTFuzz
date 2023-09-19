# adjust the gpu based on your condition
CUDA_VISIBLE_DEVICES=0 nohup python -u generate_response.py \
    --index 0 \
    --num_workers 10 \
    --model_path meta-llama/llama-2-7b-chat-hf \
    > run_logs/llama2-7b_1.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python -u generate_response.py \
    --index 1 \
    --num_workers 10 \
    --model_path meta-llama/llama-2-7b-chat-hf \
    > run_logs/llama2-7b_2.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u generate_response.py \
    --index 2 \
    --num_workers 10 \
    --model_path meta-llama/llama-2-7b-chat-hf \
    > run_logs/llama2-7b_3.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u generate_response.py \
    --index 3 \
    --num_workers 10 \
    --model_path meta-llama/llama-2-7b-chat-hf \
    > run_logs/llama2-7b_4.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u generate_response.py \
    --index 4 \
    --num_workers 10 \
    --model_path meta-llama/llama-2-7b-chat-hf \
    > run_logs/llama2-7b_5.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -u generate_response.py \
    --index 5 \
    --num_workers 10 \
    --model_path meta-llama/llama-2-7b-chat-hf \
    > run_logs/llama2-7b_6.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 nohup python -u generate_response.py \
    --index 6 \
    --num_workers 10 \
    --model_path meta-llama/llama-2-7b-chat-hf \
    > run_logs/llama2-7b_7.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 nohup python -u generate_response.py \
    --index 7 \
    --num_workers 10 \
    --model_path meta-llama/llama-2-7b-chat-hf \
    > run_logs/llama2-7b_8.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 nohup python -u generate_response.py \
    --index 8 \
    --num_workers 10 \
    --model_path meta-llama/llama-2-7b-chat-hf \
    > run_logs/llama2-7b_9.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 nohup python -u generate_response.py \
    --index 9 \
    --num_workers 10 \
    --model_path meta-llama/llama-2-7b-chat-hf \
    > run_logs/llama2-7b_10.log 2>&1 &

