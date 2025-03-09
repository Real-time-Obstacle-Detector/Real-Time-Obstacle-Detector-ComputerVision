import os

def process_dataset(labels_folder_path, images_folder_path):
    # Define indices to remove and index mapping
    remove_indices = {'0', '2', '4'}  # Indices to remove
    index_mapping = {'1': '10'}  # Indices to change

    for filename in os.listdir(labels_folder_path):
        
        label_file_path = os.path.join(labels_folder_path, filename)
        
        if os.path.isfile(label_file_path) and label_file_path.endswith(".txt"):
            with open(label_file_path, "r") as file:
                lines = file.readlines()
            
            modified_lines = []
            
            for line in lines:
                parts = line.strip().split()
                if parts[0] in remove_indices: # Skip this line if the label index is in remove_indices
                    continue
                if parts[0] in index_mapping: # Update the index using the mapping
                    parts[0] = index_mapping[parts[0]]
                
                modified_lines.append(" ".join(parts) + "\n")
            
            # If no valid lines remain, delete the label file and its image
            if not modified_lines:
                os.remove(label_file_path)
                print(f"Removed empty label file: {label_file_path}")
                
                image_file_path = os.path.join(images_folder_path, filename.replace(".txt", ".jpg"))
                if os.path.isfile(image_file_path):
                    os.remove(image_file_path)
                    print(f"Removed corresponding image file: {image_file_path}")
            else:
                # Write the updated lines back to the label file
                with open(label_file_path, "w") as file:
                    file.writelines(modified_lines)

process_dataset(
    labels_folder_path="C:\\Users\\abt\\Documents\\Real-time-obstacle-detector\\data sets\\Dynamic Object Detection\\train\\labels",
    images_folder_path="C:\\Users\\abt\\Documents\\Real-time-obstacle-detector\\data sets\\Dynamic Object Detection\\train\\images"
)

process_dataset(
    labels_folder_path="C:\\Users\\abt\\Documents\\Real-time-obstacle-detector\\data sets\\Dynamic Object Detection\\test\\labels",
    images_folder_path="C:\\Users\\abt\\Documents\\Real-time-obstacle-detector\\data sets\\Dynamic Object Detection\\test\\images"
)

process_dataset(
    labels_folder_path="C:\\Users\\abt\\Documents\\Real-time-obstacle-detector\\data sets\\Dynamic Object Detection\\valid\\labels",
    images_folder_path="C:\\Users\\abt\\Documents\\Real-time-obstacle-detector\\data sets\\Dynamic Object Detection\\valid\\images"
)
