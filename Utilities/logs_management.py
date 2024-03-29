from pandas import DataFrame
import pandas as pd
import os

def dump_data(data:DataFrame, file_path:str, file_name:str, mode="w") -> DataFrame:
   """
   Stores the data in a csv file and returns an emptied DataFrame with the same columns as "data"
   File path should be full path: any/file.csv
   """
   #Verify if file path is valid
   invalid_characters = set(file_path).intersection("<>:\"|?*")
   if len(invalid_characters) > 0:
      print("Attempted to create file with invalid characters: " + str(invalid_characters))
      for c in invalid_characters:
         file_path = file_path.replace(c, "_")
   if file_path[-1] == " ": file_path = file_path[:-1]
   if file_path[-1] == ".": file_path = file_path[:-1]

   #Verify if file exists  
   if not os.path.exists(file_path):
      os.makedirs(file_path)

   #Create file
   file_path = os.path.join(file_path, file_name)
   if mode == "a":
      data.to_csv(file_path, mode=mode, header = not os.path.exists(file_path))
   else:
      data.to_csv(file_path, mode=mode, header = True)
   return data.iloc[0:0]

def combine_logs(output_path, output_name, files_paths = []):
   collected_data = pd.DataFrame()
   for file_path in files_paths:
      #data = pd.read_csv(file_path)     #---------recently changed
      data = read_csv(file_path)
      data["Path_origin"] = [file_path for _ in range(len(data))]
      collected_data = pd.concat([collected_data, data], ignore_index=True)
   dump_data(collected_data, output_path, output_name)
   return collected_data

def find_log_files(file_name, logs_path = None):
   """
   Returns a list of paths to files with the name "file_name" in the directory and subfolders in "logs_path"
   """
   if logs_path == None: 
      logs_path = os.getcwd()
      print("Collecting logs from ", os.getcwd())
   file_path_list = []
   for root, dirs, files in os.walk(logs_path, topdown=False):
      #if dirs == []:
      for file in files:
            if file_name == file:
               file_path_list.append(os.path.join(root, file_name))
   return file_path_list


def read_csv(file_path, remove_unnamed = True):
   """
   Reads a csv file and returns a DataFrame
   """
   if remove_unnamed:
      headers = pd.read_csv(file_path, nrows=0)
      valid_columns = [col for col in headers.columns if 'Unnamed' not in col]
      df = pd.read_csv(file_path, usecols=valid_columns)
   else:
      df = pd.read_csv(file_path)
   return df