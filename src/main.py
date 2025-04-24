from config import LOG_DIR
from logging_utils import setup_logging
from rag_runner import run_rag

if __name__ == "__main__":
    setup_logging({
        "logging": {
            "log_dir": LOG_DIR,
            "log_level": "INFO"
        }
    })
    run_rag(model_provider="hf", model_name="microsoft/phi-4")