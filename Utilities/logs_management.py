from pandas import DataFrame
import os

def dump_data(data:DataFrame, file_path:str, file_name:str) -> DataFrame:
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
   data.to_csv(file_path, mode="a", index=False, header = not os.path.exists(file_path))
   return data.iloc[0:0]