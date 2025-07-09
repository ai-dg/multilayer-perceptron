from split import *
import numpy as np

class Data:
    
    # Data from split system
    dict_csv : dict
    dict_rows : dict
    dict_columns : dict
    dict_rows_20 : dict
    dict_rows_80 : dict
    
    # Data to work train process
    dict_rows_train : dict
    dict_rows_valid : dict
    dict_columns_train : dict
    dict_columns_valid : dict
    path_train : str
    path_valid : str


    # Data to calculate neuron layers

    m : int # Count of elements 
    n_features : int # Count of features (except diagnosis)
    m_valid : int
    n_features_valid : int

    epochs : int # Number of epoches
    batch_size : int # Size of batches
    learning_rate : float # Learning rate, normally 0.03

    layers : list[int] # Size of each layer

    x_train : np.ndarray # Matrix of features values (in rows)
    y_train : np.ndarray # Vector of labels (M or B)
    x_valid : np.ndarray # Matrix of validation features values
    y_valid : np.ndarray # Vector of validation labels (M or B)


    weights : dict[int, np.ndarray] # [Layer index, W_matrix]
    biases : dict[int, np.ndarray] # [Layer index, b_vector]

    z : dict[int, np.ndarray] # Weighted sum for each layer
    a : dict[int, np.ndarray] # Activations for each layer, sigmoid


def ft_initialize_values(data : Data, gradient : str):

    data.m = len(data.dict_rows_train)
    data.m_valid = len(data.dict_rows_valid)
    
    data.n_features = len(data.dict_columns_train) - 1
    data.n_features_valid = len(data.dict_columns_valid) - 1

    data.layers = [data.n_features, 24, 24, 1]

    data.epochs = 58
    data.batch_size = 8
    data.learning_rate = 0.03

    data.x_train = np.zeros((data.n_features, data.m))
    # print(data.n_features)
    # print(data.m)
    # print(len(data.x_train))
    data.y_train = np.zeros((1, data.m))

    data.x_valid = np.zeros((data.n_features_valid, data.m_valid))
    data.y_valid = np.zeros((1, data.m_valid))

    
    for i in range(1, len(data.layers)):
        n_current = data.layers[i]
        n_prev = data.layers[i - 1]

        data.weights[i] = np.random.randn(n_current, n_prev) * 0.01
        data.biases[i] = np.zeros((n_current, 1))

        if gradient == "BGD":
            data.z[i] = np.zeros((n_current, data.m))
            data.a[i] = np.zeros((n_current, data.m))
        elif gradient == "MBGD":
            data.z[i] = np.zeros((n_current, data.batch_size))
            data.a[i] = np.zeros((n_current, data.batch_size))
        elif gradient == "SGD":
            data.z[i] = np.zeros((n_current, 1))
            data.a[i] = np.zeros((n_current, 1))





def ft_fill_x_and_y_dict(columns_main : dict, rows, x_train, y_train):
    
    columns = {}
    for k, v in columns_main.items():
        if k != 0:
            columns[k - 1] = v

    for i in range(len((columns))):
        for j in range(len(rows)):
            if columns[i][j] == "M":
                x_train[i, j] = 0
            elif columns[i][j] == "B":
                x_train[i, j] = 1
            else:
                x_train[i, j] = columns[i][j]

    for i in range(len(rows)):
        if columns_main[0][i] == "M":
            y_train[0, i] = 0
        elif columns_main[0][i] == "B":
            y_train[0, i] = 1
        else:
            y_train[0, i] = 0


    


def main():

    try:

        gradient = input("Choose a method: BGD, MBGD or SGD: ")
        if gradient not in ["BGD", "MBGD", "SGD"]:
            raise ValueError

    except (ValueError, TypeError, KeyboardInterrupt):
        print("Method not available.")
        exit(1)



    data = Data()
    data.dict_rows_train = {}
    data.dict_rows_valid = {}
    data.dict_columns_train = {}
    data.dict_columns_valid = {}
    
    data.dict_csv = {}
    data.dict_rows = {}
    data.dict_columns = {}
    data.dict_rows_20 = {}
    data.dict_rows_80 = {}

    data.weights = {}
    data.biases = {}
    data.z = {}
    data.a = {}

    data.path_train = "./datasets/train.csv"
    data.path_valid = "./datasets/valid.csv"

    data.dict_columns_train = ft_import_csv_with_pandas(data.path_train)
    data.dict_rows_train = ft_convert_columns_to_rows(data.dict_columns_train)

    data.dict_columns_valid = ft_import_csv_with_pandas(data.path_valid)
    data.dict_rows_valid = ft_convert_columns_to_rows(data.dict_columns_valid)

    # print(data.dict_columns[])
    # print(len(data.dict_columns.keys()))

    ft_initialize_values(data, gradient)
    ft_fill_x_and_y_dict(data.dict_columns_train, data.dict_rows_train, data.x_train, data.y_train)
    print(data.x_train)
    print(data.y_train)
    ft_fill_x_and_y_dict(data.dict_columns_valid, data.dict_rows_valid, data.x_valid, data.y_valid)



if __name__ == "__main__":
    main()