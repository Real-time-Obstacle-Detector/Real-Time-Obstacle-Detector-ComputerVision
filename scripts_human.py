
import os
def update_labels_indices(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and file_path.endswith(".txt"):
            with open(file_path, "r") as file:
                lines = file.readlines()
            
            # Filter and modify lines
            modified_lines = []
            for line in lines:
                parts = line.strip().split()
                
                if parts[0] == '0':
                    parts[0] = '3'

                modified_lines.append(" ".join(parts) + "\n")
            
            # Write modified content back to the same file
            with open(file_path, "w") as file:
                file.writelines(modified_lines)

                            
update_labels_indices(folder_path = "C:\\Users\\abt\\Desktop\\data sets\\People.v6i.yolov8\\train\\labels" )
update_labels_indices(folder_path = "C:\\Users\\abt\\Desktop\\data sets\\People.v6i.yolov8\\test\\labels" )
update_labels_indices(folder_path = "C:\\Users\\abt\\Desktop\\data sets\\People.v6i.yolov8\\valid\\labels" )
