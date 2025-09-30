python -u code/generate_answer.py \
    --model_name_or_path=meta-llama/Llama-3.1-8B-Instruct \
    --dataset=data/separate_input.json \
    --save_path=output/vanilla \
    --gpu_id 0,1 \
    --num_gpu 2 \
    >scripts/separate_input_llama-3.1-8b-instruct_vanilla.log 2>&1



python -u code/generate_answer.py \
    --model_name_or_path=meta-llama/Llama-3.1-8B-Instruct \
    --dataset=data/coa_attack.json \
    --save_path=output/vanilla \
    --gpu_id 0,1 \
    --num_gpu 2 \
    >scripts/coa_attack_llama-3.1-8b-instruct_vanilla.log 2>&1



python -u code/generate_answer.py \
    --model_name_or_path=meta-llama/Llama-3.1-8B-Instruct \
    --dataset=data/coreference.json \
    --save_path=output/vanilla \
    --gpu_id 0,1 \
    --num_gpu 2 \
    >scripts/coreference_llama-3.1-8b-instruct_vanilla.log 2>&1



python -u code/generate_answer.py \
    --model_name_or_path=meta-llama/Llama-3.1-8B-Instruct \
    --dataset=data/error_info_answer.json \
    --save_path=output/vanilla \
    --gpu_id 0,1 \
    --num_gpu 2 \
    >scripts/error_info_answer_llama-3.1-8b-instruct_vanilla.log 2>&1

python -u code/generate_answer.py \
    --model_name_or_path=meta-llama/Llama-3.1-8B-Instruct \
    --dataset=data/fixed_task_template.json \
    --save_path=output/vanilla \
    --gpu_id 0,1 \
    --num_gpu 2 \
    >scripts/fixed_task_template_llama-3.1-8b-instruct_vanilla.log 2>&1


python -u code/generate_answer.py \
    --model_name_or_path=meta-llama/Llama-3.1-8B-Instruct \
    --dataset=data/perturbation_prompt.json \
    --save_path=output/vanilla \
    --gpu_id 0,1 \
    --num_gpu 2 \
    >scripts/perturbation_prompt_llama-3.1-8b-instruct_vanilla.log 2>&1


# Judging with three tools

python -u code/moderation_api_judge.py \
    --folder output/vanilla/Llama-3.1-8B-Instruct \                     # the folder where the LLM responses are stored
    --openai_key OPENAI_API_KEY \
    >scripts/vanilla_moderation_api_judge_Llama-3.1-8B-Instruct.log 2>&1


python -u code/llama_guard_judge.py \
    --folder output/vanilla/Llama-3.1-8B-Instruct \                     # the folder where the LLM responses are stored
    --num_gpu 2 \
    --gpu_id 0,1 \
    --openai_key OPENAI_API_KEY \
    >scripts/vanilla_judge_Llama-3.1-8B-Instruct.log 2>&1


python -u code/llm_judge.py \
    --file_fold output/vanilla/Llama-3.1-8B-Instruct \                  # the folder where the LLM responses are stored
    --openai_key OPENAI_API_KEY \
    >scripts/vanilla_llm_judge_Llama-3.1-8B-Instruct.log 2>&1


# Get the results

python -u code/majority_vote.py \
    --method vanilla \      # ['vanilla', 'cot-debias', 'self-aware', 'self-reflection', 'self-help', 'sft', 'dpo', 'task_vector', 'saferlhf']
    --output_name llama_res.json \
    >scripts/vanilla_majority_vote_Llama-3.1-8B-Instruct.log 2>&1


python code/showres.py