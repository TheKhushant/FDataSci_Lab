# multi varient == multiple Reggration (dono name same hai)
# ( df = data frame)
# (R&D Spend == research and development spend)

# code: 
import pandas as pd
df = pd.read_csv("50_Startups.csv")
df.head()
# enter 
df.info()
#enter
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#enter
