steps=${1:-3}
topk=${2:-1}
num_draft_tokens=${3:-4}
python3 benchmarks/bench_eagle3.py \
    --model-path /mnt/lm_data_afs/wangzining/charles/models/llama3-8b \
    --speculative-draft-model-path outputs/llama3-8b-eagle3-sharegpt/epoch_9_step_301690 \
    --port 30000 \
    --trust-remote-code \
    --mem-fraction-static 0.8 \
    --tp-size 1 \
    --attention-backend fa3 \
    --config-list 1,$steps,$topk,$num_draft_tokens \
    --benchmark-list mtbench gsm8k:5 ceval:5:accountant \
    --dtype bfloat16

# config-list: batch_size, steps, topk, num_draft_tokens
