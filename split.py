import sys
import csv
import numpy as np
import pandas as pd
import random

class Data:
    dict_csv : dict
    dict_rows : dict
    dict_columns : dict
    dict_rows_20 : dict
    dict_rows_80 : dict


def ft_is_float(number):
    
    try:
        float(number)
        return True
    except (TypeError, ValueError):
        return False


def ft_is_list_numeric(data : list):

    count_floats = 0
    for element in data:
        if ft_is_float(element):
            count_floats += 1

    total_elements = len(data)

    if (count_floats / total_elements) >= 90:
        return True
    else:
        return False

def ft_convert_list_to_float(data : list):

    new_list = []

    for element in data:
        try:
            new_list.append(float(element))
        except (TypeError, ValueError):
            new_list.append(0)
    
    return new_list


def ft_import_csv_with_pandas(path : str):
    
    df  = pd.read_csv(path, delimiter=',', quotechar="|", dtype=str, header=None)

    columns = {}
    # print(df.columns)

    for i, column in enumerate(df.columns):
        col_values = df[column].to_list()

        if (i != 0 and ft_is_list_numeric(col_values)):
            col_values = ft_convert_list_to_float(col_values)

        columns[i] = col_values
    
    return columns

    
def ft_convert_columns_to_rows(data : dict):
    
    columns = []
    rows = {}

    for i in sorted(data.keys()):
        columns.append(data[i])

    matrix = np.array(columns, dtype=object).T


    for i, row in enumerate(matrix):
        rows[i] = list(row)
    
    return rows


def ft_divide_rows_to_csv(data : Data):
    
    total_rows = len(data.dict_rows)
    indices = list(data.dict_rows.keys())
    seed = 42

    rows_80 = {}
    rows_20 = {}

    random.seed(seed)
    random.shuffle(indices)



    split_index = int(total_rows * 0.8)

    for i in range(total_rows):
        if i < split_index:
            rows_80[i] = data.dict_rows[indices[i]]
        else:
            rows_20[i] = data.dict_rows[indices[i]]

    
    return rows_80, rows_20
    
    


def ft_export_divided_data_to_csv(data : Data, train_path, valid_path):

    for i in data.dict_rows_20.keys():
        data.dict_rows_20[i][0], data.dict_rows_20[i][1] = data.dict_rows_20[i][1], data.dict_rows_20[i][0]
    
    for i in data.dict_rows_80.keys():
        data.dict_rows_80[i][0], data.dict_rows_80[i][1] = data.dict_rows_80[i][1], data.dict_rows_80[i][0]

    # print(data.dict_rows_20)

    with open(train_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        for i in data.dict_rows_80.keys():
            writer.writerow(data.dict_rows_80[i])

    with open(valid_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        for i in data.dict_rows_20.keys():
            writer.writerow(data.dict_rows_20[i])


def main():
    if (len(sys.argv) != 2):
        print("Wrong number of arguments")
        exit(1)

    path = sys.argv[1]
    print(f"{path}")

    data = Data()
    data.dict_csv = {}
    data.dict_rows = {}
    data.dict_columns = {}
    data.dict_rows_20 = {}
    data.dict_rows_80 = {}


    data.dict_columns = ft_import_csv_with_pandas(path)
    print(f"{len(data.dict_columns)} columns loaded.")
    data.dict_rows = ft_convert_columns_to_rows(data.dict_columns)
    data.dict_rows_80, data.dict_rows_20 = ft_divide_rows_to_csv(data)
    print(f"len 80: {len(data.dict_rows_80)}")
    print(f"len 20: {len(data.dict_rows_20)}")
    ft_export_divided_data_to_csv(data, "./datasets/train.csv", "./datasets/valid.csv")




if __name__ == "__main__":
    main()