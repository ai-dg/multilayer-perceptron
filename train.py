import json
from pipes import quote
import pandas
import sys
import csv


class Data:
    dict_csv : dict
    [...]


def ft_import_json_to_data(data: Data, path: str):
    

    with open(path, "r") as file:
        reader = csv.reader(file, delimiter=',', quotechar="|")
        index = 0
        for row in reader:
            print(f"row: {row}")
            data.dict_csv[index] = {}
            data.dict_csv[index] = row
            index =+ 1
            # for element in row:
            #     data.dict_csv[index].append(element)

    
    print(f"{data.dict_csv}")


    [...]



def main():
    if (len(sys.argv) != 2):
        print("Wrong number of arguments")
        exit(1)

    path = sys.argv[1]
    print(f"{path}")

    data = Data()
    data.dict_csv = {}

    ft_import_json_to_data(data, path)

    
    [...]





if __name__ == "__main__":
    main()