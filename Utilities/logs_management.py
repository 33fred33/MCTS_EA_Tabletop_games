from pandas import DataFrame
import os

def dump_data(self, data:DataFrame, file_path:str) -> DataFrame:
   """
   Stores the data in a csv file and returns an emptied DataFrame with the same columns as "data"
   File path should be full path: any/file.csv
   """
   data.to_csv(file_path, mode="a", header = not os.path.exists(file_path))
   return data.iloc[0:0]