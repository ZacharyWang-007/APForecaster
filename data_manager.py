from __future__ import print_function, absolute_import

import os
import glob
import os.path as osp
from datetime import datetime
import numpy as np


"""Dataset classes"""


class Climate_dataset(object):
    def __init__(self, path_indicators, path_PM25_O3, num_train=0.7, gap=7, target=None):
        
        self.path_indicators = path_indicators
        self.path_PM25_O3 = path_PM25_O3
        self.target = target
  
        # self.indicators = ['hurs', 'huss', 'pr', 'rlds', 'rsds', 'sfcWind', 'tas', 'tasmax', 'tasmin']
       
        self.files = sorted(set(os.listdir(self.path_indicators)))

        num_files = len(self.files)
        self.gap = gap
        self.num_train = int(num_files * num_train)

        self.path_pred_fire_O3 = osp.join(self.path_PM25_O3, 'pred_fire_O3')
        self.path_pred_fire_PM25 = osp.join(self.path_PM25_O3, 'pred_fire_PM25')
        self.path_pred_total_O3 = osp.join(self.path_PM25_O3, 'pred_total_O3')
        self.path_pred_total_PM25 = osp.join(self.path_PM25_O3, 'pred_total_PM25')


        train = self.files[0: self.num_train]
        val = self.files[self.num_train:]

        self.train_data = self._process_data(train, self.num_train)
        self.val_data = self._process_data(val, num_files - self.num_train)

        print("=> Dataset loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  train    | {:5d} | ".format(self.num_train))
        print("  val      | {:5d} | ".format(num_files - self.num_train))
        print("  ------------------------------")

        # breakpoint()
        self.train = train
        self.val = val


    def get_day(self, name):
        year, month, day = name.split('_')
        year = int(year)
        month = int(month)
        day = int(day.split('.')[0])

        d = datetime(year=year,month=month, day=day)
        day = d.timetuple().tm_yday
        return np.float32(np.sin(2 * np.pi * day/365) + np.cos(2 * np.pi * day/365))


    def _process_data(self, files, numbers):
        dataset = []

        for index, file in enumerate(files):
            if index + self.gap > numbers -1:
                continue
            else:  
                today, target = file, files[index + self.gap]
                target_date = target
                today, target = today.split('.')[0], target.split('.')[0]

                final = []

                final.append(osp.join(self.path_indicators, today+'.pt'))
                final.append(osp.join(self.path_indicators, target+'.pt'))

                if self.target == 'o3':
                    final.append(osp.join(self.path_pred_fire_O3, today+ '.tif'))
                    final.append(osp.join(self.path_pred_fire_O3, target+ '.tif'))
                elif self.target == 'pm25':
                    final.append(osp.join(self.path_pred_fire_PM25, today + '.tif'))
                    final.append(osp.join(self.path_pred_fire_PM25, target + '.tif'))

                final += [today, target, target_date]
                dataset.append(final)

        return dataset


if __name__ == '__main__':
    dataset = Climate_dataset()







