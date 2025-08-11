from glob import glob
import pandas as pd
import os

input_dir = "N:/data/gefs-ml/"

# Get all the files
lores_files = [f for f in glob(input_dir + '/forecast/*/*f024.csv', recursive=True) if not f.endswith('_hires.csv')]
hires_files = glob(input_dir + '/forecast/*/*f024_hires.csv', recursive=True)

# Create dictionaries to match files by their identifiers
lores_dict = {}
for f in lores_files:
    # Extract the identifier (first part when splitting on '_')
    identifier = os.path.basename(f).split('_')[0]
    lores_dict[identifier] = f

hires_dict = {}
for f in hires_files:
    # Extract the identifier (first part when splitting on '_')
    identifier = os.path.basename(f).split('_')[0]
    hires_dict[identifier] = f

# Process each matching pair of files
for identifier in lores_dict:
    # Check if there's a matching hires file
    if identifier in hires_dict:
        lores_file = lores_dict[identifier]
        hires_file = hires_dict[identifier]
        
        print(f"Processing match: {identifier}")
        print(f"  Low-res: {lores_file}")
        print(f"  High-res: {hires_file}")
        
        # Read the individual CSV files
        lores_df = pd.read_csv(lores_file)
        hires_df = pd.read_csv(hires_file)
        
        # Set index for merging
        lores_df = lores_df.set_index(['valid_datetime', 'sid'])
        hires_df = hires_df.set_index(['valid_datetime', 'sid'])
        
        # Merge prioritizing hires data
        merged_df = lores_df.combine_first(hires_df).reindex(hires_df.index.union(lores_df.index))
        
        # Create output filename
        dirname = os.path.dirname(lores_file)
        basename = os.path.basename(lores_file)
        base_without_ext = os.path.splitext(basename)[0]
        merged_filename = os.path.join(dirname, f"{base_without_ext}_merged.csv")
        
        # Save the merged data
        merged_df.to_csv(merged_filename, index=True)
        print(f"  Saved merged file: {merged_filename}")
    else:
        print(f"No high-resolution match found for: {identifier}")