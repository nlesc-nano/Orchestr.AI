import argparse
import logging

from orchestr_ai.training.training import main
from orchestr_ai.utils.helpers import load_config, parse_args
from orchestr_ai.utils.logging_utils import setup_logging

if __name__ == '__main__':
    args = parse_args()
    setup_logging()
    logging.info(f"{'*' * 30} Started {'*' * 30}")
    print("Started")
    main(args)
