import subprocess
import multiprocessing
import argparse
from tqdm import tqdm


# Function to run the script with a specific random state and dataset
def run_experiment(script_name, random_state, dataset_name, progress_bar):
    """Runs the experiment and updates the progress bar."""
    tqdm.write(f"Starting {script_name} with random_state = {random_state} and dataset = {dataset_name}")
    subprocess.run(["python3", script_name, "--random_state_list", str(random_state), "--dataset_name", dataset_name])
    tqdm.write(f"Completed {script_name} with random_state = {random_state} and dataset = {dataset_name}")
    progress_bar.update(1)  # Update progress bar


if __name__ == "__main__":
    # Command-line argument parser
    parser = argparse.ArgumentParser(description="Launch multiple experiment processes.")
    parser.add_argument(
        "experiment",
        choices=["kmeans", "random", "test_random", "test_random_pivot"],
        help="Choose which experiment to run:  'test_random', 'test_random_pivot'."
    )
    parser.add_argument(
        "dataset",
        help="Name of the dataset to use."
    )

    args = parser.parse_args()

    # Map experiment names to script filenames
    script_map = {
        "kmeans": "TEST_KMEANSCLASS_evaluation_exps.py",
        "test_random": "TEST_RANDOM_KNN_evaluation_exps.py",
        "test_random_pivot" : "TEST_RANDOM_PIVOT_evaluation_exps.py",
    }

    script_name = script_map[args.experiment]  # Get the correct script name
    dataset_name = args.dataset  # Get the dataset name from arguments

    # Define the random states
    random_state_list = [0]

    processes = []
    with tqdm(total=len(random_state_list), desc=f"Running experiments on {dataset_name}", position=0, leave=True) as pbar:
        # Create and start 10 different processes with different random states
        for random_state in random_state_list:
            p = multiprocessing.Process(target=run_experiment, args=(script_name, random_state, dataset_name, pbar))
            p.start()
            processes.append(p)

        # Wait for all processes to complete
        for p in processes:
            p.join()

    print(f"\nAll {args.experiment.upper()} experiments completed for dataset {dataset_name}!")
