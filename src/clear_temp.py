import os

# temp folders are cleared
def clear_temp():
    temp_dirs = [
        "images/temp/crops",
        "images/temp/masks",
        "images/temp/inpaintings",
        "images/temp/checkpoint"
    ]
    for dir_path in temp_dirs:
        if os.path.exists(dir_path):
            for f in os.listdir(dir_path):
                file_path = os.path.join(dir_path, f)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        else:
            os.makedirs(dir_path)
    mask_list_path = "images/temp/mask_list.txt"
    with open(mask_list_path, "w") as f:
        pass
