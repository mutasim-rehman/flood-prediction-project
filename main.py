# main.py
# ========================================================================
# This is the main, user-facing entry point for the entire project.
# It provides a simple command-line interface to choose which part of
# the pipeline to run: Data, Training, or Prediction.
#
# To use, simply run this file from your terminal:
# >> python main.py
# ========================================================================

import subprocess
import sys


def run_script(script_name):
    """Executes a given script using the same Python interpreter."""
    print(f"\n--- Running: {script_name} ---\n")
    try:
        # Using sys.executable ensures we use the same python environment
        # that is running this script.
        result = subprocess.run([sys.executable, script_name], check=True)
        print(f"\n--- Finished: {script_name} successfully. ---\n")
    except subprocess.CalledProcessError as e:
        print(f"\n--- ERROR: {script_name} failed with exit code {e.returncode}. ---")
        print("--- Please check the logs in the 'logs/' directory for details. ---")
    except FileNotFoundError:
        print(f"\n--- ERROR: Could not find the script '{script_name}'. ---")
        print("--- Please ensure you are in the project's root directory. ---")


def main_menu():
    """Displays the main menu and handles user input."""
    menu = {
        "1": ("Run the full Data Pipeline (Collector & Processor)", "main_data_pipeline.py"),
        "2": ("Train the Model", "main_train.py"),
        "3": ("Run a Prediction Demo", "main_predict.py"),
        "4": ("Exit", None)
    }

    while True:
        print("============================================")
        print(" FLOOD PREDICTION MODEL - MAIN MENU")
        print("============================================")
        print("What would you like to do?")
        for key, (description, _) in menu.items():
            print(f"  {key}) {description}")

        choice = input("\nEnter your choice (1-4): ")

        if choice in menu:
            if choice == "4":
                print("Exiting program. Goodbye!")
                break

            _, script_to_run = menu[choice]
            run_script(script_to_run)

            input("Press Enter to return to the menu...")
        else:
            print("\n*** Invalid choice. Please enter a number between 1 and 4. ***\n")


if __name__ == "__main__":
    print("RUN.PY: This is the main project runner. Please choose an option from the menu.")
    main_menu()
