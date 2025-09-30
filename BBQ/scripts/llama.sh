python -u code/vanilla.py \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --num_gpu 2 \
    --gpu_id 0,1 \
    >scripts/Llama-3.1-8B-Instruct_vanilla.log 2>&1

python -u code/self_aware.py \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --num_gpu 2 \
    --gpu_id 0,1 \
    >scripts/Llama-3.1-8B-Instruct_self_aware.log 2>&1

python -u code/self_reflection.py \
    --model_name_or_path meta-llama/Llama-3.1-8B \
    --num_gpu 2 \
    --gpu_id 0,1 \
    >scripts/Llama-3.1-8B-Instruct_self_reflection.log

python -u code/self_help.py \
    --model_name_or_path meta-llama/Llama-3.1-8B \
    --num_gpu 2 \
    --gpu_id 0,1 \
    >scripts/Llama-3.1-8B-Instruct_self_help.log

python -u code/cot_debias.py \
    --model_name_or_path meta-llama/Llama-3.1-8B \
    --num_gpu 2 \
    --gpu_id 0,1 \
    >scripts/Llama-3.1-8B-Instruct_cot_debias.log

python -u code/vanilla.py \
    --model_name_or_path SFT_LLAMA \    # the path of the model obtained after LoRA SFT
    --num_gpu 2 \
    --gpu_id 0,1 \
    >scripts/Llama-3.1-8B-Instruct_sft.log 2>&1

python -u code/vanilla.py \
    --model_name_or_path DPO_LLAMA \   # the path of the model obtained after LoRA DPO
    --num_gpu 2 \
    --gpu_id 0,1 \
    >scripts/Llama-3.1-8B-Instruct_dpo.log 2>&1

python -u code/vanilla.py \
    --model_name_or_path TASKVECTOR_LLAMA \  # the path of the model deployed Task Vector
    --num_gpu 2 \
    --gpu_id 0,1 \
    >scripts/Llama-3.1-8B-Instruct_task_vector.log 2>&1

python -u code/vanilla.py \
    --model_name_or_path RLHF_LLAMA \         # the path of the model obtained after Safe-RLHF
    --num_gpu 2 \
    --gpu_id 0,1 \
    >scripts/Llama-3.1-8B-Instruct_saferlhf.log 2>&1


python code/pre_eval_process.py --file_fold evaluation/vanilla/Llama-3.1-8B-Instruct

python -u code/majority_vote.py \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --method vanilla \  # 'cot-debias', 'self-aware', 'self-reflection', 'self-help', 'sft', 'dpo', 'task_vector', 'saferlhf'
    --output_name llama_res.json \
    >scripts/majority_vote_Llama-3.1-8B-Instruct_vanilla.log 2>&1


python code/showres.py --res results/llama_res.json