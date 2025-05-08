import logging

from mlff_qd.utils.helpers import parse_args
from mlff_qd.utils.logging_utils import setup_logging
from mlff_qd.fine_tuning.fine_tune_csvlogger import main

if __name__ == "__main__":
    args = parse_args()
    setup_logging()
    logging.info(f"{'*' * 30} Fine-tuning started {'*' * 30}")
    print("Fine-tuning started")
    main(args)

