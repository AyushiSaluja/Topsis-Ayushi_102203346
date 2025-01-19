# -*- coding: utf-8 -*-
"""assignment1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/15M8WT2ucj4N5t7Y9EOrW7ougdu4OFNgl
"""

import pandas as pd
import numpy as np

def topsis(input_file, weights, impacts, output_file):
    try:

        data = pd.read_csv("/content/102203346-data (2).csv")

        if data.shape[1] < 3:
            raise ValueError("Input file must have at least 3 columns.")

        object_names = data.iloc[:, 0]
        numeric_data = data.iloc[:, 1:]
        if not np.issubdtype(numeric_data.dtypes.values[0], np.number):
            raise ValueError("All columns except the first must contain numeric data.")

        weights = [float(w) for w in weights.split(",")]
        impacts = impacts.split(",")

        if len(weights) != numeric_data.shape[1] or len(impacts) != numeric_data.shape[1]:
            raise ValueError("Number of weights and impacts must match the number of numeric columns.")

        if not all(impact in ["+", "-"] for impact in impacts):
            raise ValueError("Impacts must be '+' or '-'.")


        norm_data = numeric_data / np.sqrt((numeric_data**2).sum(axis=0))


        weighted_data = norm_data * weights

        ideal_best = np.where(impacts == "+", weighted_data.max(axis=0), weighted_data.min(axis=0))
        ideal_worst = np.where(impacts == "+", weighted_data.min(axis=0), weighted_data.max(axis=0))

        dist_best = np.sqrt(((weighted_data - ideal_best)**2).sum(axis=1))
        dist_worst = np.sqrt(((weighted_data - ideal_worst)**2).sum(axis=1))

        topsis_score = dist_worst / (dist_best + dist_worst)

        rank = topsis_score.argsort()[::-1] + 1

        data["Topsis Score"] = topsis_score
        data["Rank"] = rank

        data.to_csv(output_file, index=False)
        print(f"Output saved to {output_file}")

    except FileNotFoundError:
        print("Error: Input file not found.")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

input_file = "102203346-data.csv"
output_file = "102203346-result.csv"
weights = "1,1,1,1,1"
impacts = "+,+,-,+,+"

topsis(input_file, weights, impacts, output_file)