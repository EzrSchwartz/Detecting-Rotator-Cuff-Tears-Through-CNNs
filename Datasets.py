import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
import tqdm 
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

class Real3DDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset = torch.load(dataset_path)  # Load .pt file

        # Ensure the dataset is structured correctly
        if isinstance(self.dataset, dict):
            self.data = self.dataset["data"]  # Assuming key is 'data'
            self.labels = self.dataset["labels"]  # Assuming key is 'labels'
        elif isinstance(self.dataset, (list, torch.Tensor)):
            self.data = self.dataset
            self.labels = None  # If no labels exist
        else:
            raise ValueError("Unsupported dataset format in .pt file.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        else:
            return self.data[idx]  # Unsupervised case


class Random3DDataset(Dataset):
    def __init__(self, num_samples=1000, depth=8, height=214, width=214, num_classes=2):
        self.num_samples = num_samples
        self.depth = depth
        self.height = height
        self.width = width
        self.num_classes = num_classes
        
        self.data = []
        self.labels = []
        
        for _ in range(num_samples):
            image = np.random.rand(depth, height, width)
            
            for d in range(depth):
                center_y = np.random.randint(0, height)
                center_x = np.random.randint(0, width)
                radius = np.random.randint(5, 15)
                
                y, x = np.ogrid[-center_y:height-center_y, -center_x:width-center_x]
                mask = x*x + y*y <= radius*radius
                image[d][mask] = np.random.rand()
                
                noise = np.random.normal(0, 0.1, (height, width))
                image[d] += noise
            
            image = (image - image.min()) / (image.max() - image.min())
            
            image_tensor = torch.FloatTensor(image).unsqueeze(0) 
            
            label = np.random.randint(0, num_classes)
            
            self.data.append(image_tensor)
            self.labels.append(label)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

if __name__ == "__Datasets__":
    # Create dataloader
    dataset = Random3DDataset()

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Example of accessing the data
    for batch_idx, (data, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}")
        print(f"Data shape: {data.shape}")  # Should be (batch_size, 1, depth, height, width)
        print(f"Labels: {labels}")
        
        if batch_idx == 0:  # Just print first batch
            break


class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_tensor = preProcessing(image_path)
        if 'Tear' in image_path:
            label = 1
        else:
            label = 0

        return image_tensor, label
class CustomDatasetUnsupervised(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_tensor = preProcessing(image_path)
        
        return image_tensor




def shoulders(Count):
    dataset = CustomDatasetUnsupervised(R"Path to shoulder data directory")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for batch_idx, (data_input) in enumerate(dataloader):
        if batch_idx == Count:
            return dataloader
def transfer(Count):
    dataset = CustomDatasetUnsupervised(R"Path to transfer data directory")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for batch_idx, (data_input) in enumerate(dataloader):
        if batch_idx == Count:
            return dataloader
def random(Count):
    dataset = Random3DDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for batch_idx, (data_input) in enumerate(dataloader):
        continue
    return dataloader

def RealData(file):
    dataset = Real3DDataset(file)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for batch_idx, (data_input) in enumerate(dataloader):
        continue
    return dataloader





# def TransferDataLoader(input_directory, output_file):
    directory = input_directory
    output_file = output_file

# List all .pt files
    pt_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pt")]

# Load and concatenate all data
    datasets = [torch.load(f) for f in pt_files]

# Check if data is in tensor format or dictionary
    if isinstance(datasets[0], dict):  # If .pt files store dictionaries
        merged_data = {key: torch.cat([d[key] for d in datasets], dim=0) for key in datasets[0].keys()}
    elif isinstance(datasets[0], torch.Tensor):  # If .pt files store tensors directly
        merged_data = torch.cat(datasets, dim=0)
    else:
        raise ValueError("Unsupported data format in .pt files.")

# Save the merged dataset
    torch.save(merged_data, output_file)
    print(f"Merged dataset saved to {output_file}")
    #make a data loader out of the merged data
    transfer_data_loader = DataLoader(merged_data, batch_size=4, shuffle=True)
    return transfer_data_loader

# def ShoulderDataLoader(input_directory, outputfile):
#     directory = input_directory
#     print("Starting ShoulderDataLoader")
#     output_file = outputfile
#     #make it go through the direcotry which has tons of .pt files and make a dataloader out of it
#     pt_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pt")]
#     print(len(pt_files))
#     datasets = [torch.load(f) for f in tqdm(pt_files)]
#     if isinstance(datasets[0], dict):  # If .pt files store dictionaries
#         merged_data = {key: torch.cat([d[key] for d in datasets], dim=0) for key in datasets[0].keys()}
#     elif isinstance(datasets[0], torch.Tensor):  # If .pt files store tensors directly
#         merged_data = torch.cat(datasets, dim=0)
#     else:
#         raise ValueError("Unsupported data format in .pt files.")
#     print("donewithgoingthroughdata")
#     torch.save(merged_data, output_file)
#     print(f"Merged dataset saved to {output_file}")
#     #make a data loader out of the merged data
#     shoulder_data_loader = DataLoader(merged_data, batch_size=4, shuffle=True)
#     return shoulder_data_loader

import torch
from torch.utils.data import Dataset, DataLoader

# Custom Dataset Class
class TensorDataset(Dataset):
    def __init__(self, tensor):
        self.tensor = tensor

    def __len__(self):
        return self.tensor.shape[0]  # Number of samples

    def __getitem__(self, idx):
        return self.tensor[idx]  # Returns a single tensor

def TransferDataLoader(output_file, batch_size=1, num_workers=4):
    # Load the merged tensor
    merged_data = torch.load(output_file)  # Ensure this is a Tensor

    # Wrap it in a dataset
    dataset = TensorDataset(merged_data)

    # Create DataLoader
    transfer_data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    return transfer_data_loader



import os
import torch
from torch.utils.data import IterableDataset, DataLoader
from typing import Iterator, Iterable, List, Dict, Union, Tuple

class ShoulderDataset(IterableDataset):
    """
    A PyTorch IterableDataset for loading shoulder MRI data from .pt files.
    The dataset handles files where both image data and labels are stored
    within the same .pt file. It supports data stored as a dictionary,
    a list, or a single tensor.  It ensures all image tensors
    are reshaped to the target size (1, 16, 214, 214) and assigns labels
    based on the filename.
    """
    def __init__(self, directory: str):
        """
        Initializes the ShoulderDataset.

        Args:
            directory: The directory containing the .pt files, where each
                       .pt file contains both image data and labels.
        """
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"Directory not found: {directory}")
        self.directory = directory
        self.pt_files = sorted([
            os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pt")
        ])
        self._length = None  # Use lazy initialization for length

    def __len__(self) -> int:
        """
        Calculates the total number of data points (images) in the dataset.
        This is calculated lazily and cached.

        Returns:
            The total number of data points in the dataset.
        """
        if self._length is None:
            total_length = 0
            for file_path in self.pt_files:
                try:
                    data = torch.load(file_path, map_location="cpu")
                    if isinstance(data, dict):
                        total_length += sum(len(v) for v in data.values() if isinstance(v, (list, torch.Tensor)))
                    elif isinstance(data, list):
                        total_length += len(data)
                    elif isinstance(data, torch.Tensor):
                        total_length += 1
                    else:
                        raise TypeError(f"Unsupported data type in {file_path}: {type(data)}")
                except Exception as e:
                    print(f"Error loading or processing {file_path} for length calculation: {e}")
            self._length = total_length
        return self._length

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Iterates over the data points in the dataset.  It loads each .pt file,
        handles the different data structures (dict, list, tensor),
        reshapes the image tensors to the target size (1, 16, 214, 214),
        extracts the corresponding labels, and yields them as a tuple.
        It also includes error handling and logging.  The label is now
        determined by the filename.

        Yields:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the image
            tensor (shape (1, 16, 214, 214)) and its corresponding label.
        """
        for file_path in self.pt_files:
            print(f"Loading {file_path}...")
            try:
                data = torch.load(file_path, map_location="cpu")  # Load data
                if "normal" in os.path.basename(file_path).lower():
                    label_value = 0
                else:
                    label_value = 1
                label = torch.tensor(label_value, dtype=torch.long) # Create the label tensor

                if isinstance(data, dict):
                    for tensor_list in data.values():
                        if isinstance(tensor_list, (list, torch.Tensor)): #check the type
                            for tensor in tensor_list:
                                processed_tensor = self._process_tensor(tensor, file_path)
                                if processed_tensor is not None: # only yield if tensor is valid
                                    yield processed_tensor, label
                elif isinstance(data, list):
                    for tensor in data:
                        processed_tensor = self._process_tensor(tensor, file_path)
                        if processed_tensor is not None: # only yield if tensor is valid
                            yield processed_tensor, label
                elif isinstance(data, torch.Tensor):
                    processed_tensor = self._process_tensor(data, file_path) #pass data
                    if processed_tensor is not None: # only yield if tensor is valid
                        yield processed_tensor, label
                else:
                    raise TypeError(f"Unsupported data format in {file_path}: {type(data)}")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

    def _process_tensor(self, tensor: torch.Tensor, file_path: str) -> Union[torch.Tensor, None]:
        """
        Helper function to process a single tensor (image data), including size
        checks and reshaping.

        Args:
            tensor: The tensor to process (image data).
            file_path: The path to the .pt file (for error reporting).

        Returns:
            Union[torch.Tensor, None]:  The processed tensor (shape (1, 16, 214, 214))
            or None if there is an error.

        Raises:
            ValueError: If the tensor has an incorrect number of elements.
            RuntimeError: If there is an error reshaping the tensor.
            TypeError: If the tensor is not a torch.Tensor
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected tensor to be torch.Tensor, got {type(tensor)} from {file_path}")

        total_elements = tensor.numel()
        expected_elements = 1 * 16 * 214 * 214
        if total_elements != expected_elements:
            print(
                f"Warning: Tensor from {file_path} has {total_elements} elements, "
                f"expected {expected_elements}. Skipping."
            )
            return None  # Skip this tensor

        try:
            processed_tensor = tensor.reshape(1, 16, 214, 214)
            return processed_tensor
        except RuntimeError as e:
            print(f"Error reshaping tensor from {file_path}: {e}")
            return None  # Skip this tensor




def ShoulderDataLoader(directory: str, batch_size: int = 9) -> DataLoader:
    """
    Creates a DataLoader from the ShoulderDataset, ensuring only complete
    batches are returned.

    Args:
        directory: The directory containing the .pt files.

    Returns:
        torch.utils.data.DataLoader: A DataLoader for the ShoulderDataset.
    """
    dataset = ShoulderDataset(directory)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        drop_last=True,
    )
    return dataloader




# Custom Dataset class for handling lists of tensors and labels
class TensorDataset(Dataset):
    def __init__(self, tensors: List[torch.Tensor], labels: List[int]):
        """
        :param tensors: List of image tensors.
        :param labels: List of corresponding numerical labels.
        """
        self.tensors = tensors
        self.labels = labels

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns the data and label for a given index.

        Args:
            idx: The index of the data point.

        Returns:
            Tuple[torch.Tensor, int]: A tuple containing the image tensor
            and its corresponding label.
        """
        return self.tensors[idx], self.labels[idx]





def create_data_loader(
    normal_dir: str,
    torn_dir: str,
    num_workers: int = 0
) -> DataLoader:
    """
    Creates a PyTorch DataLoader from the 'normal' and 'torn' image data
    saved by extractImagesR and extractImagesT.

    Args:
        normal_dir: Directory where 'normal' image tensors and labels are saved
                    (e.g., 'D:\Testing\ShoulderNormal').
        torn_dir: Directory where 'torn' image tensors and labels are saved
                    (e.g., 'D:\Testing\ShoulderTorn').
        batch_size: The batch size for the DataLoader.  Defaults to 32.
        num_workers: Number of worker processes for the DataLoader. Defaults to 0.

    Returns:
        DataLoader: A PyTorch DataLoader combining the normal and torn data.
    """
    normal_tensors = []
    normal_labels = []
    torn_tensors = []
    torn_labels = []

    # Load data from the saved .pt files.  We load the tensors and labels
    # from each seed's file, and combine them into a single list.
    for seed in range(1, len(os.listdir(normal_dir))):  # Assuming you saved files with seeds 1-5
        normal_data = torch.load(os.path.join(normal_dir, f'ShoulderNormal({seed}).pt'))
        torn_data = torch.load(os.path.join(torn_dir, f'ShoulderTorn({seed}).pt'))
        for tensor in normal_data['tensors']:
            tensor = tensor.unsqueeze(1)
        normal_tensors.extend(normal_data['tensors'])
        normal_labels.extend(normal_data['labels'])
        torn_tensors.extend(torn_data['tensors'])
        torn_labels.extend(torn_data['labels'])

    # Combine the 'normal' and 'torn' data into single lists
    all_tensors = normal_tensors + torn_tensors
    all_labels = normal_labels + torn_labels

    # Create a TensorDataset
    dataset = TensorDataset(all_tensors, all_labels)

    # Create a DataLoader
    dataloader = DataLoader(
        dataset,
        shuffle=True,  # Important to shuffle your data!
        num_workers=num_workers,
        drop_last=False  # Good to keep all data, even if last batch is smaller
    )
    return dataloader




# import torch
# from torch.utils.data import Dataset, DataLoader
# import os
# import numpy as np
# from PIL import Image

# import torch
# from torch.utils.data import Dataset, DataLoader

# class Real3DDataset(Dataset):
#     def __init__(self, dataset_path):
#         self.dataset = torch.load(dataset_path)  # Load .pt file

#         # Ensure the dataset is structured correctly
#         if isinstance(self.dataset, dict):
#             self.data = self.dataset["data"]  # Assuming key is 'data'
#             self.labels = self.dataset["labels"]  # Assuming key is 'labels'
#         elif isinstance(self.dataset, (list, torch.Tensor)):
#             self.data = self.dataset
#             self.labels = None  # If no labels exist
#         else:
#             raise ValueError("Unsupported dataset format in .pt file.")

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         if self.labels is not None:
#             return self.data[idx], self.labels[idx]
#         else:
#             return self.data[idx]  # Unsupervised case


# class Random3DDataset(Dataset):
#     def __init__(self, num_samples=1000, depth=8, height=214, width=214, num_classes=2):
#         self.num_samples = num_samples
#         self.depth = depth
#         self.height = height
#         self.width = width
#         self.num_classes = num_classes
        
#         self.data = []
#         self.labels = []
        
#         for _ in range(num_samples):
#             image = np.random.rand(depth, height, width)
            
#             for d in range(depth):
#                 center_y = np.random.randint(0, height)
#                 center_x = np.random.randint(0, width)
#                 radius = np.random.randint(5, 15)
                
#                 y, x = np.ogrid[-center_y:height-center_y, -center_x:width-center_x]
#                 mask = x*x + y*y <= radius*radius
#                 image[d][mask] = np.random.rand()
                
#                 noise = np.random.normal(0, 0.1, (height, width))
#                 image[d] += noise
            
#             image = (image - image.min()) / (image.max() - image.min())
            
#             image_tensor = torch.FloatTensor(image).unsqueeze(0) 
            
#             label = np.random.randint(0, num_classes)
            
#             self.data.append(image_tensor)
#             self.labels.append(label)

#     def __len__(self):
#         return self.num_samples

#     def __getitem__(self, idx):
#         return self.data[idx], self.labels[idx]

# if __name__ == "__Datasets__":
#     # Create dataloader
#     dataset = Random3DDataset()

#     dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
#     # Example of accessing the data
#     for batch_idx, (data, labels) in enumerate(dataloader):
#         print(f"Batch {batch_idx}")
#         print(f"Data shape: {data.shape}")  # Should be (batch_size, 1, depth, height, width)
#         print(f"Labels: {labels}")
        
#         if batch_idx == 0:  # Just print first batch
#             break


# class CustomDataset(Dataset):
#     def __init__(self, data_dir):
#         self.data_dir = data_dir
#         self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')]

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         image_path = self.image_paths[idx]
#         image_tensor = preProcessing(image_path)
#         if 'Tear' in image_path:
#             label = 1
#         else:
#             label = 0

#         return image_tensor, label
# class CustomDatasetUnsupervised(Dataset):
#     def __init__(self, data_dir):
#         self.data_dir = data_dir
#         self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')]

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         image_path = self.image_paths[idx]
#         image_tensor = preProcessing(image_path)
        
#         return image_tensor




# def shoulders(Count):
#     dataset = CustomDatasetUnsupervised(R"Path to shoulder data directory")
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
#     for batch_idx, (data_input) in enumerate(dataloader):
#         if batch_idx == Count:
#             return dataloader
# def transfer(Count):
#     dataset = CustomDatasetUnsupervised(R"Path to transfer data directory")
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
#     for batch_idx, (data_input) in enumerate(dataloader):
#         if batch_idx == Count:
#             return dataloader
# def random(Count):
#     dataset = Random3DDataset()
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
#     for batch_idx, (data_input) in enumerate(dataloader):
#         continue
#     return dataloader

# def RealData(file):
#     dataset = Real3DDataset(file)
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
#     for batch_idx, (data_input) in enumerate(dataloader):
#         continue
#     return dataloader





# def TransferDataLoader():
#     directory = "/home/ec2-user/Shoulders"
#     output_file = "/home/ec2-user/TrainingData/TransferLearingData.pt"

# # List all .pt files
#     pt_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pt")]

# # Load and concatenate all data
#     datasets = [torch.load(f) for f in pt_files]

# # Check if data is in tensor format or dictionary
#     if isinstance(datasets[0], dict):  # If .pt files store dictionaries
#         merged_data = {key: torch.cat([d[key] for d in datasets], dim=0) for key in datasets[0].keys()}
#     elif isinstance(datasets[0], torch.Tensor):  # If .pt files store tensors directly
#         merged_data = torch.cat(datasets, dim=0)
#     else:
#         raise ValueError("Unsupported data format in .pt files.")

# # Save the merged dataset
#     torch.save(merged_data, output_file)
#     print(f"Merged dataset saved to {output_file}")
#     #make a data loader out of the merged data
#     transfer_data_loader = DataLoader(merged_data, batch_size=4, shuffle=True)
#     return transfer_data_loader

# def ShoulderDataLoader():
#     directory = "/home/ec2-user/Shoulders"
#     output_file = "/home/ec2-user/TrainingData/ShoulderData.pt"
#     #make it go through the direcotry which has tons of .pt files and make a dataloader out of it
#     pt_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pt")]
#     datasets = [torch.load(f) for f in pt_files]
#     if isinstance(datasets[0], dict):  # If .pt files store dictionaries
#         merged_data = {key: torch.cat([d[key] for d in datasets], dim=0) for key in datasets[0].keys()}
#     elif isinstance(datasets[0], torch.Tensor):  # If .pt files store tensors directly
#         merged_data = torch.cat(datasets, dim=0)
#     else:
#         raise ValueError("Unsupported data format in .pt files.")
#     torch.save(merged_data, output_file)
#     print(f"Merged dataset saved to {output_file}")
#     #make a data loader out of the merged data
#     shoulder_data_loader = DataLoader(merged_data, batch_size=4, shuffle=True)
#     return shoulder_data_loader


# def create_data_loader(
#     normal_dir: str,
#     torn_dir: str,
#     num_workers: int = 0
# ) -> DataLoader:
#     """
#     Creates a PyTorch DataLoader from the 'normal' and 'torn' image data
#     saved by extractImagesR and extractImagesT.

#     Args:
#         normal_dir: Directory where 'normal' image tensors and labels are saved
#                     (e.g., 'D:\Testing\ShoulderNormal').
#         torn_dir: Directory where 'torn' image tensors and labels are saved
#                     (e.g., 'D:\Testing\ShoulderTorn').
#         batch_size: The batch size for the DataLoader.  Defaults to 32.
#         num_workers: Number of worker processes for the DataLoader. Defaults to 0.

#     Returns:
#         DataLoader: A PyTorch DataLoader combining the normal and torn data.
#     """
#     normal_tensors = []
#     normal_labels = []
#     torn_tensors = []
#     torn_labels = []

#     # Load data from the saved .pt files.  We load the tensors and labels
#     # from each seed's file, and combine them into a single list.
#     for seed in range(1, len(os.listdir(normal_dir))):  # Assuming you saved files with seeds 1-5
#         normal_data = torch.load(os.path.join(normal_dir, f'ShoulderNormal({seed}).pt'))
#         torn_data = torch.load(os.path.join(torn_dir, f'ShoulderTorn({seed}).pt'))
#         for tensor in normal_data['tensors']:
#             tensor = tensor.unsqueeze(1)
#         normal_tensors.extend(normal_data['tensors'])
#         normal_labels.extend(normal_data['labels'])
#         torn_tensors.extend(torn_data['tensors'])
#         torn_labels.extend(torn_data['labels'])

#     # Combine the 'normal' and 'torn' data into single lists
#     all_tensors = normal_tensors + torn_tensors
#     all_labels = normal_labels + torn_labels

#     # Create a TensorDataset
#     dataset = TensorDataset(all_tensors, all_labels)

#     # Create a DataLoader
#     dataloader = DataLoader(
#         dataset,
#         shuffle=True,  # Important to shuffle your data!
#         num_workers=num_workers,
#         drop_last=False  # Good to keep all data, even if last batch is smaller
#     )
#     return dataloader


