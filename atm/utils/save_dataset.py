import os
from datetime import datetime

import pandas as pd


class CSVDataset:
    def __init__(self, filepath: str, columns: list|tuple) -> None:
        self.filepath = filepath if filepath.endswith('.csv') else filepath + '.csv'
        self.columns = columns
        self.data_loaded = False
    
    def get_empty_df(self):
        self.resolve_parent_dir()
        return pd.DataFrame(columns=self.columns)

    def resolve_parent_dir(self):
        if not os.path.exists(self.filepath):
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            self.create_empty_file()

    def create_empty_file(self):
        # open(self.filepath,'w').close()
        self.get_empty_df().to_csv(self.filepath, index=False, mode='w')


    def save(self):
        # print(self.__dict__)
        try:
            self.df.to_csv(self.filepath, index=False, mode='a', header=False)
            self.clear()
        except Exception as e:
            print(e)

    def clear(self):
        print("clear called")
        self.df = self.get_empty_df()

    def parse(self, landmarks, w, h, label):
        if not self.data_loaded:
            self.df = self.get_empty_df()
            self.empty_df = self.df.copy()
            self.data_loaded = True

        record = {}
        record['datetime'] = datetime.now()
        angles = landmarks['angles']
        angle_names = angles.keys()
        # print(angle_names)

        distances = landmarks['distances']
        distance_names = distances.keys()

        for k in angle_names:
            record[k] = angles[k]['angle']
            record[k+'_start_angle'] = angles[k]['arc_angle1']
            record[k+'_end_angle'] = angles[k]['arc_angle2']

        for k in distance_names:
            record[k] = distances[k][1]

        record['img_w'] = w
        record['img_h'] = h

        record.update(landmarks['visibilities'])

        record['label'] = label
        # print(record)
        # print(record.keys())
        record_df = pd.DataFrame(record, index=[0])
        # if len(self.df) > 0:
        #     self.df = pd.concat([self.df, record_df], ignore_index=True)
        # else:
        #     self.df = record_df

        # DF should have only one record 
        self.df = pd.concat([self.empty_df, record_df], ignore_index=True)
        
        
        