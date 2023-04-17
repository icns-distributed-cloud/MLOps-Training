import pandas as pd

df = pd.read_csv('data/happycom01.csv')

columns = ["이산화탄소 센서_2", "온도센서_3", "습도센서_4", "PM1.0_5"]
print(df.values)







