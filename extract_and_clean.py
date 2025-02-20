import os
import tarfile

# Define the root directory where tar.gz files are located
root_dir = "/home/ec2-user/NDA/nda-tools/downloadcmd/packages/1235812/image03/00m"  # Change this to the actual directory

def extract_and_remove_tar(root_dir):
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".tar.gz"):
                tar_path = os.path.join(subdir, file)
                extract_path = subdir  # Extract in the same folder

                print(f"Extracting: {tar_path} ...")

                try:
                    # Open and extract the tar.gz file
                    with tarfile.open(tar_path, "r:gz") as tar:
                        tar.extractall(path=extract_path)
                    print(f"✅ Extracted: {tar_path}")

                    # Remove the tar.gz file after successful extraction
                    os.remove(tar_path)
                    print(f"🗑️ Deleted: {tar_path}")

                except Exception as e:
                    print(f"❌ Error extracting {tar_path}: {e}")

# Run the extraction function
extract_and_remove_tar(root_dir)
print("🎉 All extractions completed!")
