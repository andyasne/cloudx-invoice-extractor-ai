"""
Cloudx Invoice AI - Utility Functions
Helper functions for common tasks
"""
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List


def load_json(file_path: str) -> Dict:
    """Load JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict, file_path: str, indent: int = 2):
    """Save dictionary to JSON file"""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent)


def load_yaml(file_path: str) -> Dict:
    """Load YAML file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict, file_path: str):
    """Save dictionary to YAML file"""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False)


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def save_jsonl(data: List[Dict], file_path: str):
    """Save list of dictionaries to JSONL file"""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def count_parameters(model) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(num: int) -> str:
    """Format large numbers with commas"""
    return f"{num:,}"


def get_device() -> str:
    """Get available device (cuda or cpu)"""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def print_banner(text: str, width: int = 60):
    """Print formatted banner"""
    print("\n" + "="*width)
    print(text.center(width))
    print("="*width + "\n")


def print_section(title: str, content: Dict[str, Any], width: int = 60):
    """Print formatted section"""
    print("-"*width)
    print(title)
    print("-"*width)
    for key, value in content.items():
        print(f"  {key:.<30} {value}")
    print()


def ensure_dir(dir_path: str):
    """Ensure directory exists"""
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def get_file_size(file_path: str) -> str:
    """Get human-readable file size"""
    size = Path(file_path).stat().st_size
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"
