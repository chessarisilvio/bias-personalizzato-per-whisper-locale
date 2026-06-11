\"\"\"
Contextual correction module for Whisper transcription using a local LLM.

Loads a small GGUF model (e.g., Qwen3.5-9B-Q5_K_M.gguf) via llama_cpp
and provides a correct() function to refine transcriptions after bias application.
\"\"\"

import os
import sys
from typing import Optional

try:
    from llama_cpp import Llama
except ImportError:  # pragma: no cover
    Llama = None  # type: ignore


class ContextualCorrector:
    """Loads a GGUF model and provides contextual correction."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 2048,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
        verbose: bool = False,
    ):
        """
        Initialize the corrector with a GGUF model.

        Args:
            model_path: Path to the .gguf model file. If None, reads from
                environment variable WHISPER_CONTEXT_MODEL or defaults to
                './models/Qwen3.5-9B-Q5_K_M.gguf' relative to this file.
            n_ctx: Context size for the model.
            n_threads: Number of CPU threads to use. If None, defaults to
                number of physical cores.
            n_gpu_layers: Number of layers to offload to GPU (if supported).
            verbose: Whether to print llama_cpp internal logs.
        """
        if Llama is None:
            raise ImportError(
                "llama_cpp is not installed. Install it with: "
                "pip install --user --break-system-packages llama-cpp-python"
            )

        if model_path is None:
            model_path = os.environ.get(
                "WHISPER_CONTEXT_MODEL",
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "models",
                    "Qwen3.5-9B-Q5_K_M.gguf",
                ),
            )
        model_path = os.path.normpath(model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                "Set WHISPER_CONTEXT_MODEL environment variable or place the model in the models directory."
            )

        # Determine thread count if not provided
        if n_threads is None:
            # Try to get physical core count; fallback to os.cpu_count()
            try:
                n_threads = len(os.sched_getaffinity(0))
            except AttributeError:
                n_threads = os.cpu_count() or 1

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )

    def correct(self, text: str, max_tokens: int = 256, temperature: float = 0.2) -> str:
        """
        Apply contextual correction to a transcription string.

        Args:
            text: The transcription string (already bias-applied).
            max_tokens: Maximum number of tokens to generate for correction.
            temperature: Sampling temperature (lower = more deterministic).

        Returns:
            Corrected transcription string.
        """
        # Simple prompt: ask the model to correct transcription errors
        prompt = f"""Correct the following transcription for grammar, spelling, and clarity.
Only output the corrected text, no extra commentary.

Transcription: {text}
Corrected:"""

        # Generate response
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["\n", "Transcription:", "Corrected:"],
            echo=False,
        )

        # Extract generated text
        corrected = output["choices"][0]["text"].strip()
        return corrected if corrected else text


def correct_text(text: str) -> str:
    """
    Convenience function that loads the model (once) and corrects text.
    Uses a singleton pattern to avoid reloading the model on every call.

    Args:
        text: The transcription string to correct.

    Returns:
        Corrected transcription string.
    """
    if not hasattr(correct_text, "_corrector"):
        correct_text._corrector = ContextualCorrector()  # type: ignore
    return correct_text._corrector.correct(text)  # type: ignore


if __name__ == "__main__":  # pragma: no cover
    # Simple CLI test
    import sys

    if len(sys.argv) < 2:
        print("Usage: python contextual_correction.py \"<transcription>\"")
        sys.exit(1)

    input_text = sys.argv[1]
    print("Original:", input_text)
    print("Corrected:", correct_text(input_text))