import os
import shutil
import time

train_folder_name = 'train'
validation_folder_name = 'val'
test_folder_name = 'test'

CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))


def prepare_folders_and_data(path_src, path_dest, make_test_folder=False):
    print("DATA  FOLDER: {}".format(path_dest))
    for (dirpath, dirnames, filenames) in os.walk(path_src):
        if filenames:
            dir_name = os.path.basename(dirpath)
            prepare_dir("{}/{}".format(path_dest, dir_name))
            number_of_files = len(filenames) - 10
            leaf_type = dir_name.split("___")[0]
            condition = ''
            if "healthy" in dir_name:
                condition = "healthy"
            else:
                condition = "unhealthy"
            if make_test_folder:
                path_to_test_folder = os.path.join(CURRENT_FOLDER, "data_set", test_folder_name, dir_name)
                prepare_dir(os.path.join(CURRENT_FOLDER, "data_set", test_folder_name, dir_name))
            for cnt, file in enumerate(filenames):
                path_fo_file = os.path.join(dirpath, file)
                print("COPY FILE FROM {}".format(path_fo_file))
                renamed_file_name = "{}_{}_{}.JPG".format(leaf_type, condition, cnt)
                if cnt < number_of_files:
                    pth = os.path.join(path_dest, dir_name, renamed_file_name)
                    shutil.copyfile(path_fo_file, pth)
                else:
                    if make_test_folder:
                        pth = os.path.join(path_to_test_folder, renamed_file_name)
                        shutil.copyfile(path_fo_file, pth)
                    else:
                        pth = os.path.join(path_dest, dir_name, renamed_file_name)
                        shutil.copyfile(path_fo_file, pth)


def prepare_dir(path):
    print("MKDIR IN {}".format(path))
    try:
        time.sleep(1)
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)

prepare_folders_and_data("PlantVillage/train", os.path.join(CURRENT_FOLDER, "data_set", train_folder_name), make_test_folder=True)
prepare_folders_and_data("PlantVillage/val", os.path.join(CURRENT_FOLDER, "data_set", validation_folder_name), make_test_folder=False)
prepare_folders_and_data("PlantVillage/val", os.path.join(CURRENT_FOLDER, "data_set", test_folder_name), make_test_folder=False)
