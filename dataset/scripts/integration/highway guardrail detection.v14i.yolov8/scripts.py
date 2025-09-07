import os

def merge_and_process_dataset(labels_folder_path, images_folder_path):
    # Define labels to remove and index mappings
    remove_indices = {}
    index_mapping = {
        '0': '13',
        '1': '12'
    }
    

    for filename in os.listdir(labels_folder_path):
        label_file_path = os.path.join(labels_folder_path, filename)
        
        if os.path.isfile(label_file_path) and label_file_path.endswith(".txt"):
            with open(label_file_path, "r") as file:
                lines = file.readlines()
            
            modified_lines = []
            
            for line in lines:
                parts = line.strip().split()
                if parts[0] in remove_indices:
                        continue
                elif parts[0] in index_mapping:
                    # Update the index using the mapping
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

merge_and_process_dataset("./test/labels/", "./test/images/")
merge_and_process_dataset("./train/labels/", "./train/images/")
merge_and_process_dataset("./valid/labels/", "./valid/images/")