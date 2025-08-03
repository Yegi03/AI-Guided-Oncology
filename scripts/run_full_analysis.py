"""
main.py

This file acts as the central entry point for the 'cancer-modeling' project.
It can orchestrate various parts of the workflow, such as:
 - PDE integration (integration.py)
 - Bayesian inference / MCMC (mcmc.py)
 - Machine learning models (GBM, NN, RF) in src/models/


"""

import sys
import os

# If you need to programmatically call your other scripts, you might import them:
# from src.integration import integration
# from src.estimation import mcmc
# from src.models import gbm, nn, rf

def main():
    """
    Main workflow entry point.
    You can use command-line arguments or a simple menu to run different parts
    of the pipeline:
      1) PDE-based integration
      2) MCMC-based parameter estimation
      3) ML models (GBM, NN, RF) for predictions
      etc.
    """

    print("[main.py] Welcome to the cancer-modeling project.")
    print("[main.py] Available tasks:")
    print("  1) PDE Integration (integration.py)")
    print("  2) MCMC Parameter Estimation (mcmc.py)")
    print("  3) Train ML Models: GBM, NN, RF")

    # You could parse command-line arguments or ask for user input:
    if len(sys.argv) < 2:
        print("[main.py] Usage: python main.py <task>")
        print("[main.py] e.g., 'python main.py mcmc' to run MCMC.")
        sys.exit(0)

    task = sys.argv[1].lower()

    if task == "integration":
        print("[main.py] Running PDE integration workflow...")
        # You might call a function from integration.py, e.g.:
        # integration.main()
        # or just run `os.system("python src/integration/integration.py")`
        os.system("python src/integration/integration.py")

    elif task == "mcmc":
        print("[main.py] Running Bayesian MCMC parameter estimation...")
        # e.g. call function or run script
        # mcmc.main()  # if you imported mcmc
        os.system("python src/estimation/mcmc.py")

    elif task == "gbm":
        print("[main.py] Running Gradient Boosting Regressor example...")
        os.system("python src/models/gbm.py")

    elif task == "nn":
        print("[main.py] Running Neural Network example...")
        os.system("python src/models/nn.py")

    elif task == "rf":
        print("[main.py] Running Random Forest example...")
        os.system("python src/models/rf.py")

    else:
        print(f"[main.py] Unrecognized task: {task}")
        print("[main.py] Valid tasks: integration, mcmc, gbm, nn, rf")

    print("[main.py] Done.")

if __name__ == "__main__":
    import pandas as pd

    # Use relative paths
    input_file = "datasets/Qi70/Qi70.xlsx"
    output_file = "datasets/Qi70/75_qi70.xlsx"

    # Load the Excel file
    df = pd.read_excel(input_file)

    # Randomly sample 75% of the data
    sampled_df = df.sample(frac=0.75, random_state=42)

    # Save the sampled data to a new file
    sampled_df.to_excel(output_file, index=False)

    print(f"Sampled data saved to {output_file}")
