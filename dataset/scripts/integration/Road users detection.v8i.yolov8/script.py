import os

def merge_and_process_dataset(labels_folder_path, images_folder_path):
    # Define labels to remove and index mappings
    remove_indices = {'0', '2', '3', '7', '9', '10', '16', '17', '18', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29'}
    index_mapping = {
        '1': '2',
        '4': '5',
        '5': '14',
        '6': '3',
        '8': '5',
        '11': '5',
        '12': '5',
        '19': '7'
    }
    prefixes_to_remove = ("adit", "malam", "siang")  # Prefixes for special handling of index 18

    for filename in os.listdir(labels_folder_path):
        label_file_path = os.path.join(labels_folder_path, filename)
        
        if os.path.isfile(label_file_path) and label_file_path.endswith(".txt"):
            with open(label_file_path, "r") as file:
                lines = file.readlines()
            
            modified_lines = []
            remove_file = False  # Flag to remove the entire file if needed
            
            for line in lines:
                parts = line.strip().split()
                if parts[0] in remove_indices:
                    if parts[0] == '18' and filename.startswith(prefixes_to_remove):
                        # Remove the line if index is 18 and filename starts with specific prefixes
                        continue
                    elif parts[0] == '18':
                        # Change index 18 to 3 if filename does not match prefixes
                        parts[0] = '2'
                    else:
                        # Skip other indices in remove_indices
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