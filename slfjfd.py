import pandas as pd

# Read the original Excel file
original_df = pd.read_excel('C:/Users/kchw3_r2l9a77/OneDrive/Desktop/1_5_Waypoint_UTM_240430.txt')

# Reverse the order of rows in the first two columns
reversed_df = original_df.copy()
reversed_df.iloc[:, :2] = reversed_df.iloc[::-1, :2].values

# Write the modified data to a new Excel file
reversed_df.to_excel('C:/Users/kchw3_r2l9a77/OneDrive/Desktop/reversed_file.txt', index=False)