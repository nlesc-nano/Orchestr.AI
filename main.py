
import logging
from utils.helpers import parse_args, load_config, get_optimizer_class, get_scheduler_class
from utils.logging_utils import setup_logging
from utils.data_processing import prepare_data
from utils.model import build_model, get_task
from utils.training import train_model

@setup_logging
def main():
    args = parse_args()
    logging.info(f"{'*' * 30} Started {'*' * 30}")
    print("Started")

    # Load configuration
    config = load_config(args.config)

    # Prepare data
    custom_data, transformations = prepare_data(config)

    # Build model and task
    model = build_model(config, transformations)
    task = get_task(model, config)

    # Train the model
    train_model(config, custom_data, task)

if __name__ == '__main__':
    main()
