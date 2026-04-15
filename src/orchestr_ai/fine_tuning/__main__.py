import logging

from orchestr_ai.utils.helpers import parse_args
from orchestr_ai.utils.logging_utils import setup_logging
from orchestr_ai.fine_tuning.fine_tune import main

if __name__ == "__main__":
    args = parse_args()
    setup_logging()
    logging.info(f"{'*' * 30} Fine-tuning started {'*' * 30}")
    print("Fine-tuning started")
    main(args)

