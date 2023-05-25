import pickle as pkl
import pandas as pd
with open("twohand.pickle", "rb") as f:
    object = pkl.load(f)

df = pd.DataFrame(object)
df.to_csv(r'twohand.csv')
