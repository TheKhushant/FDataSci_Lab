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
#no Output of above code
df.State.unique()
# Output = array(['New York', 'California', 'Florida'], dtype=object)
df.State = le.fit_transform(df.State)
df.State.head()
#enter
df.corr()
#enter
	#now to visulisation
import matplotlib.pyplot as plt
import seaborn as sns
#no Output 

sns.heatmap(df)
#output  = color graph of HeatMap
#type of analysis = uni-variate, bi-variate, multi-variate

sns.heatmap(df.corr()) 
#output of chessBord like graph
#enter 

sns.heatmap(df.corr(),annot=True)
#OutPut = chessBord with values
#enter

sns.pairplot(df)
#multiple graph ke sath relation batayega
#enter

plt.boxplot(df["Marketing Spend"],vert=False)
#plt and sns bot can use here (above)
#enter

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df.columns
#output = all couumns

df.drop(columns = "Administration" , inplace = True)
#to remove columns of Administration
#inplace = True = use to confirm changes of it

df.info()
#show the info of data (check whether it is removed or not)

x = df.iloc[:,:-1]
x.head()
#output =  
#-1 = python me piche se age index me ayege to  -3 -2 -1
#	aage  se piche jaoge to 0 1 2 3 

y = df.Profit
y.head()

x = sc.fit_transform(x) 
x

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2)
xtrain

#(note = head sirf pandas me hai, numpy me nahi hai)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
#enter

# to create model below
lr.fit(xtrain,ytrain)

pred = lr.predict(xtrain)
pred
#training of it

from sklearn.metrics import r2_score
print(r2_score(ytrain,pred))
#traning of this

resultytest = lr.predict(xtest)
resultytest 
#testing of it

print(r2_score(ytest,resultytest))
#check acuraccy


print("Coeficient",lr.coef_)
print("INtercept",lr.intercept_)
#enter (coeficient and intercept value)
