"""
Evaluation metrics for Cloudx Invoice AI
Includes accuracy, precision, recall, F1, and custom invoice-specific metrics
"""
import json
from typing import Dict, List, Tuple, Any
from collections import defaultdict

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import Levenshtein


class InvoiceMetrics:
    """Calculate evaluation metrics for invoice extraction"""

    def __init__(self, fields: List[str]):
        """
        Initialize metrics calculator

        Args:
            fields: List of invoice fields to evaluate
        """
        self.fields = fields
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.ground_truths = []
        self.field_correct = defaultdict(int)
        self.field_total = defaultdict(int)
        self.exact_matches = 0
        self.total_samples = 0

    def update(self, prediction: Dict, ground_truth: Dict):
        """
        Update metrics with a new prediction

        Args:
            prediction: Predicted invoice fields
            ground_truth: Ground truth invoice fields
        """
        self.predictions.append(prediction)
        self.ground_truths.append(ground_truth)
        self.total_samples += 1

        # Check exact match
        if prediction == ground_truth:
            self.exact_matches += 1

        # Check per-field accuracy
        for field in self.fields:
            self.field_total[field] += 1

            pred_value = prediction.get(field, "")
            gt_value = ground_truth.get(field, "")

            # Exact match for field
            if pred_value == gt_value:
                self.field_correct[field] += 1

    def compute_field_accuracy(self) -> Dict[str, float]:
        """
        Compute per-field accuracy

        Returns:
            Dictionary of field -> accuracy
        """
        field_acc = {}
        for field in self.fields:
            if self.field_total[field] > 0:
                field_acc[field] = self.field_correct[field] / self.field_total[field]
            else:
                field_acc[field] = 0.0

        return field_acc

    def compute_exact_match_accuracy(self) -> float:
        """
        Compute exact match accuracy (all fields correct)

        Returns:
            Exact match accuracy
        """
        if self.total_samples == 0:
            return 0.0

        return self.exact_matches / self.total_samples

    def compute_average_field_accuracy(self) -> float:
        """
        Compute average field accuracy across all fields

        Returns:
            Average field accuracy
        """
        field_acc = self.compute_field_accuracy()
        if len(field_acc) == 0:
            return 0.0

        return sum(field_acc.values()) / len(field_acc)

    def compute_levenshtein_similarity(self) -> Dict[str, float]:
        """
        Compute Levenshtein similarity for each field

        Returns:
            Dictionary of field -> average similarity
        """
        field_similarities = defaultdict(list)

        for pred, gt in zip(self.predictions, self.ground_truths):
            for field in self.fields:
                pred_value = str(pred.get(field, ""))
                gt_value = str(gt.get(field, ""))

                if len(gt_value) == 0:
                    similarity = 1.0 if len(pred_value) == 0 else 0.0
                else:
                    distance = Levenshtein.distance(pred_value, gt_value)
                    similarity = 1.0 - (distance / max(len(pred_value), len(gt_value)))

                field_similarities[field].append(similarity)

        # Compute averages
        avg_similarities = {}
        for field, similarities in field_similarities.items():
            if len(similarities) > 0:
                avg_similarities[field] = sum(similarities) / len(similarities)
            else:
                avg_similarities[field] = 0.0

        return avg_similarities

    def compute_all_metrics(self) -> Dict[str, Any]:
        """
        Compute all metrics

        Returns:
            Dictionary containing all metrics
        """
        metrics = {
            "exact_match_accuracy": self.compute_exact_match_accuracy(),
            "average_field_accuracy": self.compute_average_field_accuracy(),
            "field_accuracy": self.compute_field_accuracy(),
            "levenshtein_similarity": self.compute_levenshtein_similarity(),
            "total_samples": self.total_samples
        }

        return metrics

    def print_metrics(self):
        """Print formatted metrics"""
        metrics = self.compute_all_metrics()

        print("\n" + "="*60)
        print("CLOUDX INVOICE AI - EVALUATION METRICS")
        print("="*60)
        print(f"Total Samples: {metrics['total_samples']}")
        print(f"\nExact Match Accuracy: {metrics['exact_match_accuracy']:.4f}")
        print(f"Average Field Accuracy: {metrics['average_field_accuracy']:.4f}")

        print("\n" + "-"*60)
        print("Per-Field Accuracy:")
        print("-"*60)
        for field, acc in sorted(metrics['field_accuracy'].items()):
            print(f"  {field:.<30} {acc:.4f}")

        print("\n" + "-"*60)
        print("Per-Field Levenshtein Similarity:")
        print("-"*60)
        for field, sim in sorted(metrics['levenshtein_similarity'].items()):
            print(f"  {field:.<30} {sim:.4f}")

        print("="*60 + "\n")


def token2json(tokens: str, processor) -> Dict:
    """
    Convert token sequence back to JSON

    Args:
        tokens: Token sequence string
        processor: Donut processor

    Returns:
        Dictionary of extracted fields
    """
    # Remove special tokens
    output = {}

    # Parse tokens
    current_field = None
    current_value = ""

    i = 0
    while i < len(tokens):
        if tokens[i:i+2] == "<s":
            # Start of field
            end_idx = tokens.find(">", i)
            if end_idx != -1:
                field_name = tokens[i+3:end_idx]
                current_field = field_name
                current_value = ""
                i = end_idx + 1
        elif tokens[i:i+3] == "</s":
            # End of field
            end_idx = tokens.find(">", i)
            if end_idx != -1:
                if current_field:
                    output[current_field] = current_value.strip()
                    current_field = None
                    current_value = ""
                i = end_idx + 1
        else:
            # Regular character
            if current_field:
                current_value += tokens[i]
            i += 1

    return output


def evaluate_predictions(
    predictions_file: str,
    ground_truth_file: str,
    fields: List[str]
) -> Dict[str, Any]:
    """
    Evaluate predictions from file

    Args:
        predictions_file: Path to predictions JSONL file
        ground_truth_file: Path to ground truth JSONL file
        fields: List of fields to evaluate

    Returns:
        Dictionary of metrics
    """
    # Load predictions
    predictions = []
    with open(predictions_file, 'r', encoding='utf-8') as f:
        for line in f:
            predictions.append(json.loads(line.strip()))

    # Load ground truths
    ground_truths = []
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        for line in f:
            ground_truths.append(json.loads(line.strip()))

    # Calculate metrics
    metrics_calculator = InvoiceMetrics(fields)

    for pred, gt in zip(predictions, ground_truths):
        metrics_calculator.update(pred, gt)

    metrics = metrics_calculator.compute_all_metrics()
    metrics_calculator.print_metrics()

    return metrics
