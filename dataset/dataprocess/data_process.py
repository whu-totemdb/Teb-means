from collections import defaultdict
import pandas as pd
import configparser
import sys
from sklearn.preprocessing import StandardScaler

def clean_data(df, config, dataset):
    selected_columns = config[dataset].getlist("columns")
    variables_of_interest = config[dataset].getlist("variable_of_interest")

    # bucketize text data
    text_columns = config[dataset].getlist("text_columns", [])
    for col in text_columns:
        df[col] = df[col].astype('category').cat.codes

    variable_columns = [df[var] for var in variables_of_interest]

    for col in df:
        if col in text_columns or col not in selected_columns:
            continue
        df[col] = df[col].astype(float)

    if config["DEFAULT"].getboolean("describe_selected"):
        print(df.describe())

    return df, variable_columns


def scale_data(df):
    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df[df.columns]), columns=df.columns)
    return df


# subsample data to max_num points
def subsample_data(df, max_num):
    return df.sample(n = max_num).reset_index(drop = True)


def take_by_key(dic, seq):
    return {k : v for k, v in dic.items() if k in seq}


def read_list(config_string, delimiter=','):
    config_list = config_string.replace("\n", "").split(delimiter)
    return [s.strip() for s in config_list]


def process(config_str, example_config):
    print("Using config_str = {}".format(config_str))
    print("Config:", example_config.sections())
    print("Config string:", config_str)
    # get parameters from example_config.ini
    data_dir = example_config[config_str].get("data_dir")
    dataset = example_config[config_str].get("dataset")
    num_clusters = list(map(int, example_config[config_str].getlist("num_clusters")))
    delta = example_config[config_str].getfloat("deltas")
    max_points = example_config[config_str].getint("max_points")
    violating = example_config["DEFAULT"].getboolean("violating")
    violation = example_config["DEFAULT"].getfloat("violation")

    # get parameters from clustering_config.ini
    clustering_config_file = example_config[config_str].get("config_file")
    dataset_config = configparser.ConfigParser(converters={'list': read_list})
    dataset_config.read(clustering_config_file)
    csv_file = dataset_config[dataset]["csv_file"]

    max_points = 100000  # max data scale

    # read the data
    df = pd.read_csv(csv_file, sep=dataset_config[dataset]["separator"])
    if max_points and len(df) > max_points:
        df = subsample_data(df, max_points)
    selected_columns1 = dataset_config[dataset].getlist("columns")
    df1 = df[[col for col in selected_columns1]].copy()
    df1 = df1.fillna(df1.median())
    df[selected_columns1] = df1
    # clean the data (bucketize text data)
    df, _ = clean_data(df, dataset_config, dataset)
    variable_of_interest = dataset_config[dataset].getlist("variable_of_interest")

    # Select only the desired columns
    selected_columns = dataset_config[dataset].getlist("columns")
    df = df[[col for col in selected_columns]]

    # Scale data if desired
    scaling = dataset_config["DEFAULT"].getboolean("scaling")
    if scaling:
        df = scale_data(df)
    df.to_csv('../output/' + dataset+'_' + str(max_points) +'_' + str(num_clusters) + '.csv', encoding="utf-8", index=False)


import csv

def process_data(input_file, output_file):
    with open(input_file, 'r') as infile:
        reader = csv.reader(infile)
        data = list(reader)

    processed_data = []
    feature_names = []
    flag = 1
    for row in data:
        if not row:
            continue

        if flag == 1:
            feature_names = ['x{}'.format(i) for i in range(len(row))]
            processed_data.append(feature_names)
            flag = 0
        else:
            processed_row = [list(map(float, item.split(":")))[1] if len(item.split(":")) == 2 else item for item in row]
            processed_data.append(processed_row)

    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(processed_data)


if __name__ == '__main__':
    config_file = "./example_config.ini"
    example_config = configparser.ConfigParser(converters={'list': read_list})
    example_config.read(config_file)

    # Create your own entry in `example_config.ini` and change this str to run
    # dataset_list = ["census1990", "hmda"]
    # dataset_list = ["UCI_Credit_Card", "adult_data"]
    # dataset_list = ["svmlight", "bank"]
    # dataset_list = ["Crime", "disease"]
    dataset_list = ["CO"]

    for dataset in dataset_list:
        process(dataset, example_config)
    
    # process_data('../rawdata/8-svmlight_host.csv', '../rawdata/processed_data.csv')