import os
import subprocess
from pandas import *


data = read_csv('pathtoCSVfile.csv')

s3_links = data['IMAGE_FILE'].tolist()
# List of S3 download links


# Directory to store the extracted files
output_dir = '/path/to/output/directory'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

for link in s3_links:
    # Extract the file name from the link
    file_name = link.split('/')[-1]
    file_path = os.path.join(output_dir, file_name)
    
    # Download the tar.gz file from S3
    subprocess.run(['aws', 's3', 'cp', link, file_path])
    
    # Create a subdirectory for the extracted contents
    extract_dir = os.path.join(output_dir, file_name.replace('.tar.gz', ''))
    os.makedirs(extract_dir, exist_ok=True)
    
    # Extract the tar.gz file
    subprocess.run(['tar', '-xzf', file_path, '-C', extract_dir])
    
    # Optionally, remove the tar.gz file after extraction
    os.remove(file_path)