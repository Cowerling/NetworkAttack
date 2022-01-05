import os
import numpy as np
import sys

from single_detect import single_detect_event
from sequence_detect import sequence_detect_event
from monitor import MonitorValue

root_dir = r'./data'

normal_data_file = os.path.join(root_dir, '正常数据.csv')
sample_data_file = os.path.join(root_dir, '样本数据.csv')

estimate_data_file = os.path.join(root_dir, '攻击数据.csv')

interval = 24
outliers_count = 5
threshold = 0.2
moment_length = 5
label_value = 1
list_min_size = 2
k1 = 0.85
k2 = 0.85
repair_size = 20
rollback = moment_length - 1
under_mean_threshold = 0.2
k3 = 0.7

single_detect_monitor_result, single_detect_mean_result = single_detect_event(sample_data_file,
                                                                              normal_data_file,
                                                                              estimate_data_file,
                                                                              interval, outliers_count, threshold, 3)
sequence_detect_day_result, sequence_detect_monitor_result, labels = sequence_detect_event(normal_data_file,
                                                                                           estimate_data_file,
                                                                                           interval, True, moment_length, 1)
print()