import os

def update_labels_indices(labels_folder_path, images_folder_path):
    for filename in os.listdir(labels_folder_path):
        label_file_path = os.path.join(labels_folder_path, filename)
        
        if os.path.isfile(label_file_path) and label_file_path.endswith(".txt"):
            with open(label_file_path, "r") as file:
                lines = file.readlines()
            
            # Filter and modify lines
            modified_lines = []
            remove_files = False  # Flag to check if both label and image files should be removed

            for line in lines:
                parts = line.strip().split()
                
                if parts[0] == '0':
                    parts[0] = '13'

                if parts[0] == '1':
                    parts[0] = '14'

                if  parts[0] == '14' or parts[0] == '3' :
                    # Set flag to remove both label and image files, then skip this line
                    remove_files = True
                    break  # Exit loop if label 10 is found, as we want to delete the entire file
                else:
                    modified_lines.append(" ".join(parts) + "\n")
                    print(parts[0])

            if remove_files:
                # Remove both label and image files if label 10 was found
                os.remove(label_file_path)
                print(f"Removed label file: {label_file_path}")
                
                image_file_path = os.path.join(images_folder_path, filename.replace(".txt", ".jpg"))
                if os.path.isfile(image_file_path):
                    os.remove(image_file_path)
                    print(f"Removed image file: {image_file_path}")
            else:
                # Write modified content back to the same label file if no label 10 was found
                with open(label_file_path, "w") as file:
                    file.writelines(modified_lines)

update_labels_indices(
    labels_folder_path = "C:\\Users\\abt\\Desktop\\data sets\\latest.v1i.yolov8\\train\\labels",
    images_folder_path= "C:\\Users\\abt\\Desktop\\data sets\\latest.v1i.yolov8\\train\\images")
update_labels_indices(
    labels_folder_path = "C:\\Users\\abt\\Desktop\\data sets\\latest.v1i.yolov8\\test\\labels",
    images_folder_path= "C:\\Users\\abt\\Desktop\\data sets\\latest.v1i.yolov8\\test\\images")
update_labels_indices(
    labels_folder_path = "C:\\Users\\abt\\Desktop\\data sets\\latest.v1i.yolov8\\valid\\labels",
    images_folder_path= "C:\\Users\\abt\\Desktop\\data sets\\latest.v1i.yolov8\\valid\\images")
