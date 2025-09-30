





# BiasFreeBench: a Benchmark for Mitigating Bias in Large Language Model Responses


<p align="center">
  <a href="">📃 Paper</a> 
  <a href="https://github.com/xxupiano/BiasFreeBench">💻 Code</a> 
</p>



<div align=center><img src="assets/overview.png" width="100%"/></div>


**BiasFreeBench** is an empirical benchmark that comprehensively compares eight mainstream bias mitigation techniques (covering four prompting-based and four training-based methods) on two test scenarios (multi-choice QA and open-ended multi-turn QA) by reorganizing existing datasets into a unified query-response setting. We hope that this benchmark can serve as a unified testbed for bias mitigation methods.

<br><br>
## 📌 Table of Contents

- [🛠️ Setup](#1)
- [💬 Prompting-based Methods and Evaluation](#2)
    
- [💻 Training-based Methods](#3)
- [📝 Citation](#4)
- [✨ Acknowledgements](#5)

<br><br>

<h2 id="1">🛠️ Setup</h2>

```bash
conda create -n biasfree python=3.12 -y && conda activate biasfree
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
cd ..
pip install vllm==0.8.5 deepspeed==0.15.4 pandas wandb
```

<br><br>


<h2 id="2">💬 Prompting-based Methods and Evaluation</h2>

<div align=center><img src="assets/prompting.png" width="100%"/></div>

- Some example scripts, covering all debiasing methods explored in BiasFreeBench, for Llama-3.1-8B-Instruct is in `BBQ/scripts/llama.sh` and `FairMT-Bench/scripts/llama.sh`
- When using a reasoning LLM, please set `--max_output_len`, `--temperature`, `--top_p`, `--top_k`, and `--min_p` as the official suggestions of the corresponding models. For example, for Qwen3-8B, we set `--max_output_len 32768 --temperature 0.6 --top_p 0.95 --top_k 20 --min_p 0` based on the sampling parameters suggested in the [Qwen-3B model card](https://huggingface.co/Qwen/Qwen3-8B).

<br><br>

<h2 id="3">💻 Training-based Methods</h2>

<div align=center><img src="assets/Method.png" width="100%"/></div>



Following [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), we provide the data and scripts.

```bash
cp Training/data/* LLaMA-Factory/data
cp -r Training/examples/debias LLaMA-Factory/examples
cp -r Training/scripts LLaMA-Factory/scripts
cd LLaMA-Factory
```

<h3>SFT and DPO</h3>

Configurations are in `examples/debias`. There are example scripts for Llama-3.1-8B-Instruct in `scripts/llama.sh`

<h3>Task Vector</h3>

- Full SFT with LLaMA-Factory and the example script `examples/debias/debias_full_sft_llama_tv.yaml`
- `cd Task_Vector`
- `git clone https://github.com/mlfoundations/task_vectors.git`
- `cp * task_vectors/src`, `cd task_vectors/src`
- Modify the model paths in `test.sh` and then run `bash test.sh`

<h3>Safe Alignment</h3>

Follow [Safe RLHF](https://github.com/PKU-Alignment/safe-rlhf) with the 2 training stages:

  
1. Value Models (reward model & cost model): `--model_name_or_path` is a path/model name from HuggingFace of the instruction-tuned models, such as [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) and  [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct). 
2. Safe-RLHF: `--actor_model_name_or_path` is a path/model name from HuggingFace of the instruction-tuned models.

An example of commands to run the whole pipeline with Llama-3.1-8B-Instruct is as follows:
```bash
git clone https://github.com/PKU-Alignment/safe-rlhf.git
cd safe-rlhf
conda env create --file conda-recipe.yaml
conda activate safe-rlhf
bash scripts/reward-model.sh --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --output_dir output/rm
bash scripts/cost-model.sh --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --output_dir output/cm
bash scripts/ppo-lag.sh \
    --actor_model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --reward_model_name_or_path output/rm \
    --cost_model_name_or_path output/cm \
    --output_dir output/ppo-lag
```

<br><br>

<h2 id="4">📝 Citation</h2>

```bibtex
@article{biasfreebench25,
    title={BiasFreeBench: a Benchmark for Mitigating Bias in Large Language Model Responses},
    author={Xin Xu, Xunzhi He, Churan Zhi, Ruizhe Chen, Julian McAuley, Zexue He},
    year={2025}
}
```
<br><br>

<h2 id="5">✨ Acknowledgements</h5>

- Thanks for the code from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and [Safe RLHF](https://github.com/PKU-Alignment/safe-rlhf).
- Thanks for the data from [BBQ](https://github.com/nyu-mll/BBQ/tree/main/data) and [FairMT-Bench](https://github.com/FanZT6/FairMT-bench/tree/main/FairMT_10K).
