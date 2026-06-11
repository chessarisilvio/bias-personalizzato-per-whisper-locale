\"\"\"
Bias loader and applier for Whisper transcription correction.

Supports loading bias definitions from JSON or TSV files.
JSON format: {"original": "replacement", ...}
TSV format: two columns (original\\treplacement) without header.
\"\"\"

import json
import os
from typing import Dict, Optional


class BiasLoader:
    """Load bias mappings from a file."""

    def __init__(self, bias_file: Optional[str] = None):
        """
        Initialize BiasLoader.

        Args:
            bias_file: Path to bias file (JSON or TSV). If None, looks for
                       'bias.json' in the data directory relative to this module.
        """
        if bias_file is None:
            # Default to data/bias.json relative to this file's location
            base_dir = os.path.dirname(__file__)
            bias_file = os.path.join(base_dir, '..', 'data', 'bias.json')
        self.bias_file = os.path.normpath(bias_file)
        self.mappings: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        """Load bias mappings from the file."""
        if not os.path.exists(self.bias_file):
            raise FileNotFoundError(f"Bias file not found: {self.bias_file}")

        ext = os.path.splitext(self.bias_file)[1].lower()
        if ext == '.json':
            self._load_json()
        elif ext in ('.tsv', '.csv', '.txt'):
            self._load_tsv()
        else:
            raise ValueError(f"Unsupported bias file extension: {ext}")

    def _load_json(self) -> None:
        """Load bias mappings from a JSON file."""
        with open(self.bias_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("JSON bias file must contain a dictionary mapping")
        # Ensure all keys and values are strings
        self.mappings = {str(k): str(v) for k, v in data.items()}

    def _load_tsv(self) -> None:
        """Load bias mappings from a TSV/CSV file (two columns, no header)."""
        self.mappings = {}
        with open(self.bias_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.rstrip('\\n\\r')
                if not line:
                    continue
                parts = line.split('\\t')
                if len(parts) != 2:
                    # Try comma as fallback
                    parts = line.split(',')
                if len(parts) != 2:
                    raise ValueError(
                        f"Line {line_num} in bias file must have exactly two columns: {line}"
                    )
                original, replacement = parts
                self.mappings[original.strip()] = replacement.strip()

    def get_mappings(self) -> Dict[str, str]:
        """Return a copy of the bias mappings."""
        return self.mappings.copy()


def apply_bias(transcript: str, bias_loader: Optional[BiasLoader] = None) -> str:
    """
    Apply bias corrections to a transcript.

    Replaces occurrences of bias keys with their corresponding values.
    Replacement is done in a single pass, longest keys first to avoid
    partial overlaps.

    Args:
        transcript: The raw transcription string from Whisper.
        bias_loader: An optional BiasLoader instance. If None, a default
                     loader is instantiated (which will look for data/bias.json).

    Returns:
        The corrected transcription string.
    """
    if bias_loader is None:
        bias_loader = BiasLoader()

    mappings = bias_loader.get_mappings()
    if not mappings:
        return transcript

    # Sort keys by length descending to replace longer matches first
    sorted_keys = sorted(mappings.keys(), key=len, reverse=True)

    result = transcript
    for key in sorted_keys:
        if key:
            result = result.replace(key, mappings[key])
    return result


if __name__ == '__main__':
    # Simple self-test when run directly
    import sys
    test_file = sys.argv[1] if len(sys.argv) > 1 else None
    loader = BiasLoader(test_file) if test_file else BiasLoader()
    print(f"Loaded {len(loader.get_mappings())} bias entries")
    # Demo
    demo = "Hello world, this is a test."
    print(f"Original: {demo}")
    print(f"Corrected: {apply_bias(demo, loader)}")