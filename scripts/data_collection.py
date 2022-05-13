import os
import glob
import chardet
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class data_collection():

    def data_fetch_parse(path: str) -> dict:
        # Get CSV files list from a folder
        csv_files = glob.glob(path + "/*.csv")
        csv_files_name = [file for file in os.listdir(
            path) if file.endswith(".csv")]
        # Read each CSV file into DataFrame
        # This creates a list of dataframes
        df_list = {}
        for file, file_name in zip(csv_files, csv_files_name):
            # Look at the first ten thousand bytes to guess the character encoding with confidence interval
            with open(file, 'rb') as rawdata:
                encod_type = chardet.detect(rawdata.read(10000))["encoding"]
                # parse data with given file path
                df = pd.read_csv(file, sep=None, delim_whitespace=None,
                                 encoding=encod_type, engine="python")
                if set(["SK_ID_CURR"]).issubset(df.columns):
                    df = df.drop("SK_ID_CURR", axis=1)
                    df_list[file_name] = df
                else:
                    df_list[file_name] = df

        return df_list
