# Efficient dLLM Inference

```bash
module load python/3.12.5-fasrc01 cuda/11.8.0-fasrc01 cudnn/8.9.2.26_cuda11-fasrc01
conda activate rtx
bash setup.sh
bash reproduce.sh --workflow paper --max_samples 50
bash reproduce.sh --workflow train --stage warmstart
bash reproduce.sh --workflow train --stage grpo
bash reproduce.sh --workflow eval --dataset gsm8k --checkpoint auto
bash reproduce.sh --workflow eval --dataset math500 --checkpoint auto
bash reproduce.sh --workflow eval --dataset humaneval --checkpoint auto
```