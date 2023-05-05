from pandas import DataFrame
import os

def dump_data(data:DataFrame, file_path:str, file_name:str) -> DataFrame:
   """
   Stores the data in a csv file and returns an emptied DataFrame with the same columns as "data"
   File path should be full path: any/file.csv
   """
   if not os.path.exists(file_path):
      os.makedirs(file_path)
   file_path = os.path.join(file_path, file_name)
   data.to_csv(file_path, mode="a", header = not os.path.exists(file_path))
   return data.iloc[0:0]