import glob
import os
import shutil
import zipfile

dir_name = "ADAM_release_subjs"
temp_dir_path = "_extracted_data_temp"
images_dir_path = "images"
labels_dir_path = "labels"

zip_files_path = glob.glob("{}/*.zip".format(dir_name))
for zip_file_path in zip_files_path:
    zip_f = zipfile.ZipFile(zip_file_path)
    zip_f.extractall(temp_dir_path)
    zip_f.close()

images = glob.glob("{}/*/orig/TOF.nii.gz".format(temp_dir_path))
labels = glob.glob("{}/*/aneurysms.nii.gz".format(temp_dir_path))

if not os.path.exists(images_dir_path):
    os.makedirs(images_dir_path)

if not os.path.exists(labels_dir_path):
    os.makedirs(labels_dir_path)

for image_path, label_path in zip(images, labels):
    new_name = "{}.nii.gz".format(image_path.split("/")[1])
    shutil.copyfile(image_path, "images/{}".format(new_name))
    shutil.copyfile(label_path, "labels/{}".format(new_name))

shutil.rmtree(temp_dir_path)
