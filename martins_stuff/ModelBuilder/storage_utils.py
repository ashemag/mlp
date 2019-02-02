import pickle
import os
import csv

def save_statistics(statistics_to_save,file_path):
    """
    :param statistics_to_save: dict, val type is float
    :param file_path: e.g. file_path = "C:/test_storage_utils/dir2/test.txt"
    :return:
    """

    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_path,'a+') as f: # append mode + creates if doesn't exist
        header = ""
        line = ""

        keys = ["current_epoch","train_acc","train_loss","epoch_train_time"] # ordered in this way.

        for i,key in enumerate(keys):
            val = statistics_to_save[key]

            if i==0:
                line = line + "{:.4f}".format(val)
                header = header + key
            else:
                line = line + "\t" + "{:.4f}".format(val)
                header = header + "\t" + key

        if os.stat(file_path).st_size == 0:  # if empty
            f.write(header+"\n")
        f.write(line+"\n")

def main():
    pass

if __name__ == '__main__':
    main()