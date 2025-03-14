from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import argparse
import os

def extract_metrics_data(event_file):
    """
    Extracts metrics data (scalars) from a TensorBoard event file and returns it as a Pandas DataFrame.
    """
    try:
        event_acc = EventAccumulator(event_file)
        event_acc.Reload()
        tags = event_acc.Tags().get('scalars', [])
        if not tags:
            return pd.DataFrame(columns=['Step', 'Metric', 'Value'])
        metrics_data = []
        for tag in tags:
            events = event_acc.Scalars(tag)
            for event in events:
                metrics_data.append([event.step, tag, event.value])
        df = pd.DataFrame(metrics_data, columns=['Step', 'Metric', 'Value'])
        return df
    except Exception as e:
        print(f"Error processing event file {event_file}: {e}")
        return pd.DataFrame(columns=['Step', 'Metric', 'Value'])

def parse_command_line_arguments():
    """Parses command line arguments for the script."""
    parser = argparse.ArgumentParser(description='Extract metrics data from a TensorBoard event file and save to CSV in "results" folder.')
    parser.add_argument('-p', '--path', type=str, required=True, help='Path to the TensorBoard event file') # -p for path
    parser.add_argument('-o', '--output_file', type=str, default='metrics_output.csv', help='Base name for the output CSV file (will be saved in "results" folder, default: metrics_output.csv)') # -o for output_file
    return parser.parse_args()

def save_metrics_to_csv(df, output_csv_file):
    """Saves the metrics DataFrame to a CSV file and handles potential errors."""
    if not df.empty:
        try:
            df.to_csv(output_csv_file, index=False)
            print(f"Metrics data saved to CSV file: {output_csv_file}")
        except Exception as e:
            print(f"Error saving CSV file: {e}")
    else:
        print("No metrics data extracted from the event file. CSV not saved.")

def main():
    """Main function to execute the script."""
    args = parse_command_line_arguments()
    event_file = args.path
    output_base_filename = args.output_file

    results_folder = os.path.join("analysis", "results")  # Results folder INSIDE analysis

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        print(f"Created folder: {results_folder}")

    output_csv_file = os.path.join(results_folder, output_base_filename)

    if not os.path.exists(event_file):
        print(f"Error: Event file not found at path: {event_file}")
        return  # Exit if event file is not found

    print(f"Processing event file: {event_file}")
    metrics_df = extract_metrics_data(event_file)
    save_metrics_to_csv(metrics_df, output_csv_file)

if __name__ == "__main__":
    main()