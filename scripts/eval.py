import os 
import sys
import logging
import argparse
import warnings

from cloneval import ClonEval


warnings.simplefilter(action='ignore', category=FutureWarning)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original_dir", 
        type=str,
        default="example/original_samples",
        help="Full path to the directory with original samples."
    )
    parser.add_argument(
        "--cloned_dir", 
        type=str,
        default="example/cloned_samples",
        help="Full path to the directory with cloned samples."
    )
    parser.add_argument(
        "--use_emotion", 
        action="store_true",
        help="Whether to aggregate results per emotion."
    )
    return parser.parse_args()

def check_directory(directory: str) -> bool:
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")
    if not os.listdir(directory):
        raise ValueError(f"Directory '{directory}' is empty.")
    return True

def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    if not check_directory(args.original_dir) or not check_directory(args.cloned_dir):
        sys.exit(1)
    original_dir_files = set(os.listdir(args.original_dir))
    cloned_dir_files = set(os.listdir(args.cloned_dir))
    if original_dir_files != cloned_dir_files:
        raise RuntimeError("Filenames in directories do not match.")
    logger.info(f'{args=}')
    return args

def main():
    args = parse_args()
    args = validate_args(args)

    cloneval = ClonEval()
    cloneval.evaluate(
        orig_dir=args.original_dir,
        clon_dir=args.cloned_dir,
        use_emotion=args.use_emotion,
    )


if __name__ == "__main__":
    main()