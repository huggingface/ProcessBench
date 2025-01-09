"""
Example code to evaluate the models:

python run_eval_prm_trl.py \
    --config "gsm8k" \
    --model_name "plaguss/Qwen2.5-0.5B-Math-Shepherd-PRM-0.2" \
    --output_dir "./outputs" \
    --batch_size 64 \
    --sep "\n"

python run_eval_prm_trl.py \
    --config "all" \
    --model_name "plaguss/Qwen2.5-Math-7B-Instruct-PRM-0.1" \
    --output_dir "./outputs" \
    --batch_size 64 \
    --sep "\n\n"

python run_eval_prm_trl.py \
    --config "all" \
    --model_name "plaguss/Qwen2.5-Math-1.5B-Instruct-PRM-0.2" \
    --output_dir "./outputs" \
    --batch_size 256 \
    --sep "\n\n"
"""

import argparse
import os
import json
import importlib.metadata
from pathlib import Path
from dataclasses import dataclass
import json
from functools import cached_property

from tqdm import tqdm
import numpy as np
from datasets import load_dataset, Dataset
from transformers import pipeline, Pipeline

# Check if HF_TRANSFER is available and use it
if "hf_transfer" in set(p.metadata["name"] for p in importlib.metadata.distributions()):
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


CONFIGS = ["gsm8k", "math", "olympiadbench", "omnimath"]


def f1_score(precision: float, recall: float) -> float:
    if (precision is None) or (recall is None):
        return 0
    return 2 * precision * recall / (precision + recall)


@dataclass
class Example:
    problem: str
    steps: list[str]
    label: int
    sep: str = "\n"

    @cached_property
    def get_texts(self):
        """Returns the lists with each problem and solution steps concatenated
        with the separator. 
        """
        return [
            self.sep.join((self.problem, *self.steps[:i])) + self.sep
            for i, step in enumerate(self.steps, start=1)
        ]


class BatchProcessor:
    """Helper class to allow passing batches to the model pipeline including different
    problem and solutions steps. It allows assigning back the steps of the errors at the
    end by finding the corresponding index of the problems in the batches.
    """
    def __init__(self, data: list[Example], batch_size: int = 32):
        self.data = data
        self.batch_size = batch_size
        self.current_idx = 0

        # Create index mapping for steps
        self.step_mapping = []  # [(dataset_idx, step_idx), ...]
        for idx, item in enumerate(data):
            for step_idx in range(len(item.steps)):
                self.step_mapping.append((idx, step_idx))

        self.total_steps = len(self.step_mapping)

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= self.total_steps:
            raise StopIteration

        batch_indices = []
        batch_steps = []
        step_count = 0

        while self.current_idx < self.total_steps and step_count < self.batch_size:
            dataset_idx, step_idx = self.step_mapping[self.current_idx]
            batch_indices.append((dataset_idx, step_idx))

            # Here the steps have to be already generated
            steps = self.data[dataset_idx].get_texts
            batch_steps.append(steps[step_idx])

            step_count += 1
            self.current_idx += 1

        return batch_steps, batch_indices

    def get_total_batches(self):
        """Return the total number of batches."""
        return (self.total_steps + self.batch_size - 1) // self.batch_size


def process_results(
    results: list[dict[str, bool | str | int]],
    batch_indices: list[tuple[int, int]],
    processed_data: dict[int, list[dict[str, str | float | int]]]
) -> None:
    """
    Assign results back to the original dataset structure.

    Args:
        results: List of results from processing the batch
        batch_indices: List of (dataset_idx, step_idx) tuples
        processed_data: Dictionary to store results, keyed by dataset index
    """
    for result, (dataset_idx, step_idx) in zip(results, batch_indices):
        if dataset_idx not in processed_data:
            processed_data[dataset_idx] = []
        # Ensure the list is long enough to insert at step_idx
        while len(processed_data[dataset_idx]) <= step_idx:
            processed_data[dataset_idx].append(None)
        processed_data[dataset_idx][step_idx] = result


def get_prediction_index(
    outputs: list[dict[str, str | float | int]], label_for_true: str = "LABEL_1"
) -> int:
    """Obtain the first index for which the step is considered an error, or -1.
    Finds the coincidences and the indices of the element in the batch, corresponding to the step index.
    """
    for i, output in enumerate(outputs):
        # TODO: This should be updated to account a different token as separator (for example
        # if \n\n counts as 2 tokens)
        # TODO: This can be used that use different separators:
        # encoded_input = tokenizer("\n\n", return_tensors="pt")
        # token_ids = encoded_input["input_ids"][0]
        # tokens = tokenizer.convert_ids_to_tokens(token_ids)
        # With this info, we would find the corresponding tokens in the text, and obtain the score/prediction

        prediction = True if output[-1]["entity"] == label_for_true else False
        if prediction is False:
            return i
    return -1


def obtain_results(
    examples: list[Example],
    processed_data: dict[int, list[dict[str, str | float | int]]],
) -> list[dict[str, bool | str | int]]:
    """Find the indices of the error/correct results. """
    results = []
    for i, example in enumerate(examples):
        outputs = processed_data[i]
        idx = get_prediction_index(outputs)
        results.append(
            {
                "match": idx == example.label,
                "label": example.label,
                "predicted_index": idx,
            }
        )
    return results


def precision_recall(results: list[dict[str, bool | str | int]]) -> tuple[float] | None:
    """Compute precision and recall over the list of obtained results. """
    errors = []
    corrects = []
    for result in results:
        if result["label"] == -1:
            corrects.append(result["match"])
        else:
            errors.append(result["match"])

    precision = np.mean(errors) * 100 if len(errors) > 0 else None
    recall = np.mean(corrects) * 100 if len(corrects) > 0 else None
    return precision, recall


def read_results(path: Path) -> list[dict[str, bool | str | int]]:
    """Read the already generated results and prepare the data structure. """
    results = []
    with path.open() as file:
        for line in file:
            results.append(json.loads(line))
    return results


def results_report(aggregated_results: dict[str, dict[str, int | float]]) -> None:
    """Prints the final results. """
    print("Individual Results:")
    print("-" * 70)
    max_config_length = max(len(config) for config in aggregated_results.keys())

    for config, metrics in aggregated_results.items():
        print(f"{config:<{max_config_length}} -> Precision: {metrics['precision']:.2f}  Recall: {metrics['recall']:.2f}  F1 Score: {metrics['f1_score']:.2f}")

    # Calculate weighted averages
    total_problems = sum(metrics['num_problems'] for metrics in aggregated_results.values())
    weighted_precision = 0
    weighted_recall = 0
    weighted_f1 = 0

    for metrics in aggregated_results.values():
        weight = metrics['num_problems'] / total_problems
        weighted_precision += metrics['precision'] * weight
        weighted_recall += metrics['recall'] * weight
        weighted_f1 += metrics['f1_score'] * weight

    # Print aggregated results
    print("Weighted Averages:")
    print("-" * 70)
    print(f"{'Weighted':<{max_config_length}} -> Precision: {weighted_precision:.2f}  Recall: {weighted_recall:.2f}  F1 Score: {weighted_f1:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="all",
        choices=["gsm8k", "math", "olympiadbench", "omnimath", "all"],
        help="The configuration to run from the dataset, by default will use 'all'.",
    )
    parser.add_argument("--model_name", type=str, required=True, help="")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="The path to save the results to.",
    )
    parser.add_argument(
        "--sep",
        type=str,
        default="\n",
        help="Separator of the model, ensure it corresponds to the same one used during training.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help=(
            "The number of examples to run in a single batch. Each question has multiple steps, "
            "and a batch can contain multiple from different questions to speed up the process."
        ),
    )
    parser.add_argument(
        "--max_elements",
        type=int,
        default=-1,
        help="Number of elements to run. Helpful for testing, by default will run the full dataset.",
    )

    args = parser.parse_args()

    # Determine the configs to evaluate
    configs = CONFIGS if args.config == "all" else [args.config]
    pipe: "Pipeline" | None = None

    path = Path(args.output_dir).absolute() / args.model_name.replace("/", "__")

    path.mkdir(exist_ok=True, parents=True)
    aggregated_results = {}
    for config in tqdm(configs, total=len(configs), desc="Configuration"):
        config_file = path / f"{config}.jsonl"
        if config_file.exists():
            print(f"The results already exist for {config_file}")
            results = read_results(config_file)
            num_problems = len(results)
            precision, recall = precision_recall(results)
            f1 = f1_score(precision, recall)
            aggregated_results[config] = {
                "num_problems": num_problems,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
            continue

        # Only download the model and run it if the results are not already available.
        if pipe is None:
            pipe = pipeline("token-classification", model=args.model_name, device="cuda")

        print(f"Start configuration: {config}")
        subset = load_dataset("Qwen/ProcessBench", split=config)
        if args.max_elements > -1:
            subset = subset.select(range(args.max_elements))

        # Prepare examples
        examples = [
            Example(
                problem=row["problem"],
                steps=row["steps"],
                label=row["label"],
                sep=args.sep,
            )
            for row in subset
        ]

        # Create batch processor and the data structure to store results
        batch_processor = BatchProcessor(examples, batch_size=args.batch_size)
        processed_data = {}

        for batch_steps, batch_indices in tqdm(
            batch_processor,
            total=batch_processor.get_total_batches(),
            desc="Processing batches...",
        ):
            # Actual predictions
            batched_outputs = pipe(batch_steps)
            # Assign results back to original structure
            process_results(batched_outputs, batch_indices, processed_data)

        results = obtain_results(examples, processed_data)
        num_problems = len(results)
        precision, recall = precision_recall(results)
        f1 = f1_score(precision, recall)

        aggregated_results[config] = {
            "num_problems": num_problems,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        print(f"Writing results to {config_file}")
        with open(str(config_file), "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

    results_report(aggregated_results)


if __name__ == "__main__":
    main()
