import os

def update_houses_labels(labels_folder_path):
    for filename in os.listdir(labels_folder_path):
        label_file_path = os.path.join(labels_folder_path, filename)
        
        if os.path.isfile(label_file_path) and label_file_path.endswith(".txt"):
            with open(label_file_path, "r") as file:
                lines = file.readlines()
            
            modified_lines = []
            
            for line in lines:
                parts = line.strip().split()
                if parts[0] == '0':
                    parts[0] = '1'  # Change label 0 to 12 for our data base
                
                modified_lines.append(" ".join(parts) + "\n")
            
            # Write the updated lines back to the label file
            with open(label_file_path, "w") as file:
                file.writelines(modified_lines)

            print(f"Updated labels in file: {label_file_path}")

update_houses_labels(
    labels_folder_path="C:\\Users\\abt\\Documents\\Real-time-obstacle-detector\\data sets\\AML Homework 1.v2i.yolov8\\train\\labels"
)

update_houses_labels(
    labels_folder_path="C:\\Users\\abt\\Documents\\Real-time-obstacle-detector\\data sets\\AML Homework 1.v2i.yolov8\\test\\labels"
)

update_houses_labels(
    labels_folder_path="C:\\Users\\abt\\Documents\\Real-time-obstacle-detector\\data sets\\AML Homework 1.v2i.yolov8\\valid\\labels"
)