# ProcessBench

📄 [**[paper]**](https://huggingface.co/papers/2412.06559) 🤗 [**[data]**](https://huggingface.co/datasets/Qwen/ProcessBench)

This is the official repository for paper "**ProcessBench: Identifying Process Errors in Mathematical Reasoning**"

If you find this work relevant or helpful to your work, please kindly cite us:

```
@article{processbench,
  title={ProcessBench: Identifying Process Errors in Mathematical Reasoning}, 
  author={
    Chujie Zheng and Zhenru Zhang and Beichen Zhang and Runji Lin and Keming Lu and
    Bowen Yu and Dayiheng Liu and Jingren Zhou and Junyang Lin
  },
  journal={arXiv preprint arXiv:2412.06559},
  year={2024}
}
```

## News

* **[12/13/2024]** Released the evaluation [code](./code/run_eval_prm_rlhflow.py) for the RLHFlow PRMs
* **[12/11/2024]** Released the evaluation [**code**](./code) and the [**data**](https://huggingface.co/datasets/Qwen/ProcessBench) on HuggingFace 
* **[12/10/2024]** Released the [**paper**](https://huggingface.co/papers/2412.06559) on arXiv

## Data Usage

You can use the following code to preview the ProcessBench data:

```python
import json
from datasets import load_dataset

dataset = load_dataset('Qwen/ProcessBench', split='gsm8k')
print(json.dumps(dataset[0], indent=2))

# Expected output:
"""
{
  "id": "gsm8k-0",
  "generator": "Qwen2-7B-Instruct",
  "problem": "Sue lives in a fun neighborhood...",
  "steps": [
    "To find out how many more pink plastic flamingos were out than...",
    ...
  ],
  "final_answer_correct": false,
  "label": 1
}
"""
```

## Evaluation

You can refer to the [code](./code) folder for the evaluation code and the prompt templates we use in this work


### Evaluating TRL based models

In TRL v0.13.0 a [PRM trainer](https://huggingface.co/docs/trl/v0.13.0/en/prm_trainer) was introduced. The resulting PRM returns probabilities for the different tokens, and works directly with the [Token classification](https://huggingface.co/docs/transformers/tasks/token_classification) pipeline. To evaluate these models, clone this repository and install the `requirements-trl.txt` dependencies:

```bash
uv pip install -r requirements-trl.txt
```

Now go to the `/code` folder, and run the following script:

```bash
python run_eval_prm_trl.py \
    --model_name "plaguss/Qwen2.5-Math-1.5B-Instruct-PRM-0.2" \
    --output_dir "./outputs" \
    --batch_size 256 \
    --sep "\n\n"
```

Other than the model to evaluate, and the token used as a separator, the only relevant argument is the batch size. Internally, the process runs using a transformers pipeline, and it benefits from bigger sizes. *For reference, for a 7B model, a batch size of 128 should work, taking close to 2 hours to complete the benchmark*. The results are saved in the `output_dir`, and if the command is rerun, it will check for the results to only compute the final metrics.

The help for the script can be found here:

```bash
usage: run_eval_prm_trl.py [-h] [--config {gsm8k,math,olympiadbench,omnimath,all}] --model_name MODEL_NAME [--output_dir OUTPUT_DIR] [--sep SEP] [--batch_size BATCH_SIZE] [--max_elements MAX_ELEMENTS]

options:
  -h, --help            show this help message and exit
  --config {gsm8k,math,olympiadbench,omnimath,all}
                        The configuration to run from the dataset, by default will use 'all'.
  --model_name MODEL_NAME
  --output_dir OUTPUT_DIR
                        The path to save the results to.
  --sep SEP             Separator of the model, ensure it corresponds to the same one used during training.
  --batch_size BATCH_SIZE
                        The number of examples to run in a single batch. Each question has multiple steps, and a batch can contain multiple from different questions to speed up the process.
  --max_elements MAX_ELEMENTS
                        Number of elements to run. Helpful for testing, by default will run the full dataset.
```


* Analyzing the results:

The following output corresponds to [plaguss/Qwen2.5-Math-7B-Instruct-PRM-0.2](https://huggingface.co/plaguss/Qwen2.5-Math-1.5B-Instruct-PRM-0.2):

```bash
Individual Results:
----------------------------------------------------------------------
gsm8k         -> Precision: 22.71  Recall: 93.78  F1 Score: 36.56
math          -> Precision: 38.22  Recall: 70.69  F1 Score: 49.61
olympiadbench -> Precision: 27.08  Recall: 53.98  F1 Score: 36.07
omnimath      -> Precision: 27.93  Recall: 54.77  F1 Score: 37.00
Weighted Averages:
----------------------------------------------------------------------
Weighted      -> Precision: 30.09  Recall: 63.81  F1 Score: 40.38
```

It yields the individual results, and finally the weighted average by the number of examples in in subset. The weighted F1 Score corresponds to the value shown in the reference paper to compare.
