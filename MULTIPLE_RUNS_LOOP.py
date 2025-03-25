import subprocess
import multiprocessing
import argparse
import os
from tqdm import tqdm


# Function to run the script with a specific random state and dataset name
def run_experiment(script_name, random_state, dataset_name):
    """Runs the experiment synchronously."""
    tqdm.write(f"Starting {script_name} with random_state = {random_state} and dataset = {dataset_name}")
    subprocess.run(["python3", script_name, "--random_state_list", str(random_state), "--dataset_name", dataset_name])
    tqdm.write(f"Completed {script_name} with random_state = {random_state} and dataset = {dataset_name}")


if __name__ == "__main__":
    # Command-line argument parser
    parser = argparse.ArgumentParser(description="Launch multiple experiments with different random seeds for each dataset.")
    parser.add_argument(
        "experiment",
        choices=["kmeans", "random", "test_random", "test_kmeans"],
        help="Choose which experiment to run: 'kmeans', or 'random'. Choose 'test_random'  or 'test_kmeans' for testing"
    )

    args = parser.parse_args()

    # Map experiment names to script filenames
    script_map = {
        "kmeans": "KMEANSCLASS_evaluation_exps.py",
        "random": "RANDOM_evaluation_exps.py",
        "test_random" : "TEST_RANDOM_KNN_evaluation_exps.py"
    }

    script_name = script_map[args.experiment]  # Get the correct script name

    # Define the random states
    if script_name in ["test_random", "test_kmeans"]:
        random_state_list = [0]
    else:
        random_state_list = list(range(10))  # [0, 1, 2, ..., 9]

    # Get all dataset names from the folder
    dataset_folder = "sensitivity_datasets/"
    dataset_files = [f for f in os.listdir(dataset_folder) if f.endswith(".csv")]
    dataset_names = [os.path.splitext(f)[0] for f in dataset_files]  # Remove .csv extension
    dataset_names = (dataset_names)
   

    for dataset_name in dataset_names:
        tqdm.write(f"\nProcessing dataset: {dataset_name}")

        # Create a progress bar for the dataset
        with tqdm(total=len(random_state_list), desc=f"Running {dataset_name}", position=0, leave=True) as pbar:
            processes = []
            for random_state in random_state_list:
                p = multiprocessing.Process(target=run_experiment, args=(script_name, random_state, dataset_name))
                p.start()
                processes.append(p)

            # Wait for all random states to complete before moving to the next dataset
            for p in processes:
                p.join()
                pbar.update(1)

    print(f"\nAll {args.experiment.upper()} experiments completed!")
