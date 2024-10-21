import pandas as pd
df = pd.read_csv("Salary_Data.csv")
df
//this will show the file
df.head()       // display first 5 element
df.shape		// display the row and column
df.info()  	// show data type and name of column


// for correlation 
df.corr()		// 0.97


//visulise the data using this 2 library
import matplotlib.pyplot as plt
import seaborn as sns


plt.scatter(df["Salary"],df.YearsExperience)   // display only the dot chart using scatter


// to get label title x and y in dot chart

plt.scatter(df["Salary"],df.YearsExperience)
plt.title("Salary vs tears of experience")
plt.ylabel("Salary")
plt.xlabel("Years of Experience")
plt.show()


// using of heatmap
sns.heatmap(df.corr())			// show the number of column and rows with colour 


@)  delete that column and store it in x
x = df.drop("Salary",axis=1)
x.head()			//show x

y = df['Salary']		//transfer salary into y
y.head()			//show y 





// for model train test (to create a model)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2) 		//change output continuously 
lr.fit(xtrain,ytrain)
xtrain.head()


xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2, random_state=10)   //this will fix the data for train and test
lr.fit(xtrain,ytrain)
xtrain.head()


lr.fit(xtrain,ytrain)			// it will show that our model is ready


pred = lr.predict(xtrain)		// to predict the data and get the Output// known data
pred


from sklearn.metrics import r2_score			//to check / test the value
print(r2_score(pred,ytrain))				


predytest = lr.predict(xtest)				//to predict the data and get the Output  // unknown data
predytest

print(r2_score(predytest,ytest))			//to check / test the value


