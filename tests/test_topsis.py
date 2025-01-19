# Test cases for TOPSIS implementation 
from topsis.topsis import topsis

# Example test for the topsis function
if _name_ == "_main_":
    input_file = "test-data.csv"  # Replace with a valid CSV file path
    output_file = "test-results.csv"  # Replace with a desired output path
    weights = "1,1,1,2"  # Example weights
    impacts = "+,+,-,+"  # Example impacts

    # Run the TOPSIS function
    topsis(input_file, weights, impacts, output_file)