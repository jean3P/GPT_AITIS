# src/main.py

from config import LOG_DIR
from logging_utils import setup_logging
from rag_runner import run_rag, run_batch_rag
import argparse
import logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Get available prompt names for the help message
    available_prompts = ", ".join([
        "standard", "detailed", "precise"
    ])

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run RAG system for insurance policy analysis")
    parser.add_argument("--model", choices=["openai", "hf"], default="hf", help="Model provider (openai or hf)")
    parser.add_argument("--model-name", default="microsoft/phi-4", help="Name of the model to use")
    parser.add_argument("--batch", action="store_true", help="Process all policies in a single batch")
    parser.add_argument("--num-questions", type=int, default=None,
                        help="Number of questions to process (default: all available questions)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save JSON output files (default: resources/results/json_output)")
    parser.add_argument("--prompt", default="standard",
                        help=f"Prompt template to use. Available: {available_prompts}")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        default="INFO", help="Set the logging level")
    parser.add_argument("--persona", action="store_true", help="Use persona extraction for queries")
    parser.add_argument("--questions", type=str,
                        help="Comma-separated list of question IDs to process (e.g., '1,2,3,4')")
    parser.add_argument("--policy-id", type=str,
                        help="Process only a specific policy ID (e.g., '20')")
    args = parser.parse_args()

    # Set up logging with the specified log level
    setup_logging({
        "logging": {
            "log_dir": LOG_DIR,
            "log_level": args.log_level
        }
    })

    logger.info(f"Starting RAG pipeline with model: {args.model}/{args.model_name}")
    logger.info(f"Using prompt template: {args.prompt}")
    logger.info(f"Log level set to: {args.log_level}")
    if args.num_questions:
        logger.info(f"Processing {args.num_questions} questions")
    else:
        logger.info("Processing all available questions")

    # Parse the question_ids from the arguments
    question_ids = None
    if args.questions:
        question_ids = [q.strip() for q in args.questions.split(',')]
        logger.info(f"Will process specific questions with IDs: {', '.join(question_ids)}")

    if args.policy_id:
        logger.info(f"Will only process policy with ID: {args.policy_id}")

    # Run the appropriate RAG function with all parameters
    if args.batch:
        run_batch_rag(
            model_provider=args.model,
            model_name=args.model_name,
            max_questions=args.num_questions,
            output_dir=args.output_dir,
            prompt_name=args.prompt,
            use_persona=args.persona,
            question_ids=question_ids,
            policy_id=args.policy_id
        )
    else:
        run_rag(
            model_provider=args.model,
            model_name=args.model_name,
            max_questions=args.num_questions,
            output_dir=args.output_dir,
            prompt_name=args.prompt,
            use_persona=args.persona,
            question_ids=question_ids,
            policy_id=args.policy_id
        )

    logger.info("RAG pipeline completed successfully")
