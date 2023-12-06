import os
import pandas as pd

directory = os.path.dirname(os.path.abspath('__file__'))
os.chdir(directory)
filename = "cleaned_southeast" 
csv_path = os.path.join(directory, filename)
df = pd.read_parquet(filename) # will read faster

df_clean = df[['temp', 'tmax', 'tmin', 'date_time']].dropna().reset_index(drop=True)
df_clean = df_clean[(df_clean['date_time'] >= '2000-01-01 00:00:00') & (df_clean['date_time'] <= '2021-02-02 00:00:00')]
df_clean['date_time'] = pd.to_datetime(df_clean['date_time'])
df_clean.set_index('date_time', inplace=True)

columns_to_average = ['temp', 'tmax', 'tmin']
df_daily_mean = df_clean[columns_to_average].resample('D').mean()
df_daily_mean.reset_index(inplace=True)

output_filename = "southeast_daily_mean"
df_daily_mean.to_parquet(output_filename, engine='pyarrow', compression='snappy', index=False, use_deprecated_int96_timestamps=True)