import json



def read_json_file(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def display_multiple_images(row, num_images=5, grid_size=(1, 5)):
    """
    Reshapes a row of values into multiple 8x8 grayscale images and displays them in a grid.

    Args:
        row (pd.Series): A pandas Series containing the values representing the pixel intensities.
                           Assumes the row contains enough values for num_images * 64 pixels.
        num_images (int): The number of 8x8 images to create from the row.
        grid_size (tuple): The layout of the grid (rows, cols) for displaying the images.
    """
    total_pixels = num_images * 64
    if len(row) < total_pixels:
        raise ValueError(f"Row must contain at least {total_pixels} values to create {num_images} images.")

    # Convert the row to a NumPy array
    row_values = row.to_numpy()

    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(grid_size[1] * 3, grid_size[0] * 3))  # Adjust figure size as needed

    # If only one image, axes is not an array
    if num_images == 1:
        axes = [axes]

    # Iterate through the desired number of images
    for i in range(num_images):
        # Extract the pixel data for the current image
        start_pixel = i * 64
        end_pixel = start_pixel + 64
        image_data = row_values[start_pixel:end_pixel].reshape(8, 8)

        # Clip values to be within the valid grayscale range [0, 255]
        image_data = np.clip(image_data, 0, 255)

        # Determine the correct subplot to use
        ax = axes[i]

        # Display the image on the subplot
        ax.imshow(image_data, cmap='gray', vmin=0, vmax=255)
        ax.axis('off')  # Turn off axis labels and ticks

    plt.tight_layout()  # Adjust subplot parameters for a tight layout
    plt.show()


def perform_grouped_analysis(df, groupby_column, sum_columns, mean_std_columns,target_colname):
    """
    Performs a grouped analysis on a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        groupby_column (str): The name of the column to group by.
        sum_columns (list): A list of column names to sum into a single column.
        mean_std_columns (list): A list of column names to calculate the mean and standard deviation for each group.

    Returns:
        pd.DataFrame: A DataFrame containing the results of the grouped analysis.
    """

    # 1. Sum the specified columns
    df[target_colname] = df[sum_columns].sum(axis=1)
    mean_std_columns.append(target_colname)
    # 2. Group by the specified column and calculate the sum, mean, and standard deviation
    grouped = df.groupby(groupby_column).agg(

        **{f'{col}_mean': (col, 'mean') for col in mean_std_columns},
        **{f'{col}_std': (col, 'std') for col in mean_std_columns}
    )

    return grouped