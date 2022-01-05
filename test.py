import os
import pandas as pd
import matplotlib.pyplot as plt

from filtering import wt_filter


root_dir = 'data'
file_name = '攻击数据.xlsx'
file_path = os.path.join(root_dir, file_name)

df = pd.read_excel(file_path)
data = df.iloc[:, 13].to_numpy()[1:].tolist()
datarec = wt_filter(data, 10)
# Create wavelet object and define parameters


index = [x for x in range(0, len(data))]

plt.subplot(2, 1, 1)
plt.plot(index, data)
plt.xlabel('time (s)')
plt.ylabel('microvolts (uV)')
plt.title("Raw signal")
plt.subplot(2, 1, 2)
plt.plot(index, wt_filter(data, 0.04))
plt.xlabel('time (s)')
plt.ylabel('microvolts (uV)')
plt.title("De-noised signal using wavelet techniques")

plt.tight_layout()
plt.show()