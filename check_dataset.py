import os

dataset_path = "Dataset/archive/Quality Dataset"

folders = ["train", "valid", "test"]

for folder in folders:
    print("\n" + folder.upper())

    folder_path = os.path.join(dataset_path, folder)

    for class_name in ["fresh", "rotten"]:
        class_path = os.path.join(folder_path, class_name)

        if os.path.exists(class_path):
            images = os.listdir(class_path)
            print(class_name, ":", len(images), "images")
        else:
            print(class_name, ": NOT FOUND")