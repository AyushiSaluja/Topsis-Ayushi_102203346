# TOPSIS implementation 
import pandas as pd
import numpy as np

def topsis(input_file, weights, impacts, output_file):
    try:
        # Load the input file
        data = pd.read_csv(input_file)

        # Check for at least 3 columns
        if data.shape[1] < 3:
            raise ValueError("Input file must have at least 3 columns.")

        # Extract object names and numeric data
        object_names = data.iloc[:, 0]  # First column
        numeric_data = data.iloc[:, 1:]  # Remaining columns

        # Validate numeric data
        if not np.issubdtype(numeric_data.dtypes.values[0], np.number):
            raise ValueError("All columns except the first must contain numeric data.")

        # Convert weights and impacts to lists
        weights = [float(w) for w in weights.split(",")]
        impacts = impacts.split(",")

        # Validate the number of weights and impacts
        if len(weights) != numeric_data.shape[1] or len(impacts) != numeric_data.shape[1]:
            raise ValueError("Number of weights and impacts must match the number of numeric columns.")

        if not all(impact in ["+", "-"] for impact in impacts):
            raise ValueError("Impacts must be '+' or '-'.")

        # Normalize the data
        norm_data = numeric_data / np.sqrt((numeric_data**2).sum(axis=0))

        # Apply weights
        weighted_data = norm_data * weights

        # Calculate ideal best and ideal worst
        ideal_best = np.where(np.array(impacts) == "+", weighted_data.max(axis=0), weighted_data.min(axis=0))
        ideal_worst = np.where(np.array(impacts) == "+", weighted_data.min(axis=0), weighted_data.max(axis=0))

        # Compute distances to ideal best and worst
        dist_best = np.sqrt(((weighted_data - ideal_best)**2).sum(axis=1))
        dist_worst = np.sqrt(((weighted_data - ideal_worst)**2).sum(axis=1))

        # Calculate TOPSIS score
        topsis_score = dist_worst / (dist_best + dist_worst)

        # Rank the scores
        rank = topsis_score.argsort()[::-1] + 1

        # Add scores and ranks to the original data
        data["Topsis Score"] = topsis_score
        data["Rank"] = rank

        # Save the result
        data.to_csv(output_file, index=False)
        print(f"Output saved to {output_file}")

    except FileNotFoundError:
        print("Error: Input file not found.")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")