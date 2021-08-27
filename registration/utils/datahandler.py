import pathlib
import json
from os import listdir
from os.path import isfile, isdir, join
from shutil import copyfile


class Datahandler:
    """
    Class to handle the unprocessed and preprocessed files. Creates the expected folder hierarchy for the results.
    """

    def __init__(self, name: str):
        """
        :param name: should be element of {'brain_t1', 'brain_t2', 'inter_modal_t1t2', 'neck_brain_cancer', 'home'}
        """

        self.name = name

    def get_data_path(self, name: str):
        """
        :param name: {'brain_t1', 'brain_t2', 'inter_modal_t1t2', 'neck_brain_cancer', 'home'}
        :return: path to the original data given the corresponding name
        """

        root_path = pathlib.Path().resolve()
        paths = join(root_path,'utils/paths.json2')

        with open(paths) as f:
            my_dict = json.load(f)

        return my_dict[name]

    def get_unprocessed_folder(self):
        """
        :return: path to unprocessed original data folder
        """

        if self.name == 'inter_modal_t1t2':
            return [self.get_data_path('brain_t1'), self.get_data_path('brain_t2')]

        path = self.get_data_path(self.name)

        if self.name == "neck_brain_cancer":
            path = self.restructure_neck_brain_cancer(path)

        return path

    def get_processed_folder(self):
        """
        :return: path to preprocessed data folder
        """

        path = self.get_data_path("home")

        if self.name == 'brain_t1':
            path = join(path, 'T1w/results')
        elif self.name == 'brain_t2':
            path = join(path, 'T2w/results')
        elif self.name == "neck_brain_cancer":
            path = join(path, 'neck_brain_cancer/results')
        elif self.name == "inter_modal_t1t2":
            path = self.create_t1t2()
        else:
            path = self.get_unprocessed_folder()

        return path

    def get_processed_home(self):
        """
        :return: path to base folder for corresponding case
        """

        path = self.get_data_path("home")

        if self.name == 'brain_t1':
            path = join(path, 'T1w')
        elif self.name == 'brain_t2':
            path = join(path, 'T2w')
        elif self.name == "neck_brain_cancer":
            path = join(path, 'neck_brain_cancer')
        elif self.name == "inter_modal_t1t2":
            path = join(path, 'T1_T2')
        else:
            print("test images")
            path = self.get_unprocessed_folder()
            path = join(path, 'results')
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        return path

    def get_unprocessed_file_names(self):
        """
        :return: list of all unprocessed original data files
        """

        path = self.get_unprocessed_folder()

        if self.name == "neck_brain_cancer":
            path = self.restructure_neck_brain_cancer(path)
        elif self.name == 'inter_modal_t1t2':

            files_t1 = [join(path[0], f) for f in listdir(path[0]) if isfile(join(path[0], f))]
            files_t2 = [join(path[1], f) for f in listdir(path[1]) if isfile(join(path[1], f))]

            return files_t1 + files_t2

        return [join(path, f) for f in listdir(path) if isfile(join(path, f))]

    def get_processed_file_names(self):
        """
        :return: List of preprocessed files
        """

        if self.name == "inter_modal_t1t2":
            return self.files_t1t2()

        path = self.get_processed_folder()

        try:
            files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
        except FileNotFoundError as e:
            raise Exception(path, "preprocess files first", format(e)) from None

        return files

    def restructure_neck_brain_cancer(self, path):
        """
        neck brain cancer data folder consists of a folder for each scan containing the actual MRI and the
        corresponding segmentation. In order to be able to further process the data, all these MRI scans need to be
        transfered into one folder.

        :param path: path to original data
        :return: path to a folder that contains all *nii.gz files
        """

        files = [join(path, f) for f in listdir(path) if isdir(join(path, f))]

        destination = self.get_data_path("home")

        destination_imgs = join(destination, 'neck_brain_cancer/raw')

        destination_msks = join(destination, 'neck_brain_cancer/masks')

        pathlib.Path(destination_imgs).mkdir(parents=True, exist_ok=True)
        pathlib.Path(destination_msks).mkdir(parents=True, exist_ok=True)

        for item in files:

            f = join(item, "image.nii.gz")

            new_name = join(destination_imgs, item[68:82] + ".nii.gz")

            if isfile(f) and not isfile(new_name):
                copyfile(f, new_name)

        return destination_imgs

    def create_t1t2(self):
        """
        creates the folder structure for the inter-modal T1w-T2w case
        :return: path to results of inter-modal T1w-T2w
        """

        path = self.get_data_path("home")

        path_t1t2 = join(path, 'T1_T2/results')

        pathlib.Path(path_t1t2).mkdir(parents=True, exist_ok=True)

        return path_t1t2

    def files_t1t2(self):
        """
        collects the preprocessed files from T1w and T2w images into one list
        :return: list of preprocessed files
        """

        path = self.get_data_path("home")

        path_t1 = join(path, 'T1w/results')

        path_t2 = join(path, 'T2w/results')

        path_t1t2 = join(path, 'T1_T2/results')

        pathlib.Path(path_t1t2).mkdir(parents=True, exist_ok=True)

        try:
            files_t1 = [join(path_t1, f) for f in listdir(path_t1) if isfile(join(path_t1, f))]
        except FileNotFoundError as e:
            raise Exception(path, "preprocess T1 files first", format(e)) from None

        try:
            files_t2 = [join(path_t2, f) for f in listdir(path_t2) if isfile(join(path_t2, f))]
        except FileNotFoundError as e:
            raise Exception(path, "preprocess T2 files first", format(e)) from None

        return files_t1 + files_t2


if __name__ == '__main__':
    dh_t1 = Datahandler('brain_t1')
    dh_t1t2 = Datahandler('inter_modal_t1t2')

    print(dh_t1t2.get_processed_folder())
    print(dh_t1t2.get_processed_file_names())
