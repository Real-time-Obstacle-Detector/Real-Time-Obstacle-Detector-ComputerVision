
import os

def update_labels_indices(train_folder, test_folder, validation_folder):
    # Function to update indices 0 to 17 and 1 to 18 in all files in given folders
    def process_folder(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) and file_path.endswith(".txt"):
                with open(file_path, "r") as file:
                    lines = file.readlines()
                with open(file_path, "w") as file:
                    for line in lines:
                        parts = line.strip().split()
                        # Check each part and update if it is '0' or '1'
                        for i, part in enumerate(parts):
                            if part.isdigit():  # Check if the part is a number
                                if int(part) == 0:
                                    parts[i] = '17'
                                elif int(part) == 1:
                                    parts[i] = '10'
                        file.write(" ".join(parts) + "\n")

    # Process each folder
    for folder in [train_folder, test_folder, validation_folder]:
        process_folder(folder)

#update_labels_indices(
#    train_folder = "C:\\Users\\abt\\Desktop\\data sets\\walls finder.v1i.yolov8\\train\\labels" ,
#    test_folder = "C:\\Users\\abt\\Desktop\\data sets\\walls finder.v1i.yolov8\\test\\labels" ,
#    validation_folder = "C:\\Users\\abt\\Desktop\\data sets\\walls finder.v1i.yolov8\\valid\\labels" 
#)

def remove_lables(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and file_path.endswith(".txt"):
            with open(file_path, "r") as file:
                lines = file.readlines()
            
            # Filter and modify lines
            modified_lines = []
            for line in lines:
                parts = line.strip().split()

                if parts[0] != '17':
                    modified_lines.append(" ".join(parts) + "\n")
                    print(parts[0])
                    continue
                else:
                    print(parts[0],"removing unsueful data set index")
            
            # Write modified content back to the same file
            with open(file_path, "w") as file:
                file.writelines(modified_lines)

#remove_lables(folder_path = "C:\\Users\\abt\\Desktop\\data sets\\walls finder.v1i.yolov8\\train\\labels" )
#remove_lables(folder_path = "C:\\Users\\abt\\Desktop\\data sets\\walls finder.v1i.yolov8\\test\\labels" )
#remove_lables(folder_path = "C:\\Users\\abt\\Desktop\\data sets\\walls finder.v1i.yolov8\\valid\\labels" )
