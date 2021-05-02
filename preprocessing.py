import warnings
warnings.filterwarnings("ignore")

import math
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score,KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error,mean_squared_error

df=pd.read_csv(r'C:\Users\chand\Desktop\hackathon\all_matches.csv')

df['wicket_type']=np.where(df['wicket_type'].isnull(),0,1)
df['other_wicket_type']=np.where(df['other_wicket_type'].isnull(),0,1)
df['total_wickets']=df['wicket_type']+df['other_wicket_type']
df['total_runs']=df['runs_off_bat']+df['extras']

df=df.drop(['match_id','season','wides','noballs','byes','legbyes','penalty','non_striker','player_dismissed','other_wicket_type','other_player_dismissed','wicket_type','runs_off_bat','extras'],axis=1)

df['batting_team']=np.where(df['batting_team']=='Delhi Daredevils','Delhi Capitals',df['batting_team'])
df['bowling_team']=np.where(df['bowling_team']=='Delhi Daredevils','Delhi Capitals',df['bowling_team'])
df['batting_team']=np.where(df['batting_team']=='Kings XI Punjab','Punjab Kings',df['batting_team'])
df['bowling_team']=np.where(df['bowling_team']=='Kings XI Punjab','Punjab Kings',df['bowling_team'])


current_team=['Kolkata Knight Riders', 'Royal Challengers Bangalore',
       'Chennai Super Kings', 'Punjab Kings', 'Rajasthan Royals','Mumbai Indians','Sunrisers Hyderabad',
        'Delhi Capitals']
df=df[df['batting_team'].isin(current_team)]
df=df[df['bowling_team'].isin(current_team)]

df=df[df['ball']<6.1]

current_stadium=['Eden Gardens','Wankhede Stadium','M Chinnaswamy Stadium','MA Chidambaram Stadium','Arun Jaitley Stadium','Narendra Modi Stadium']

chidambaram_stadium=['MA Chidambaram Stadium, Chepauk, Chennai','MA Chidambaram Stadium, Chepauk','MA Chidambaram Stadium']
chinnaswamy_stadium=['M Chinnaswamy Stadium','M.Chinnaswamy Stadium']

df['venue']=np.where(df['venue']=='Wankhede Stadium, Mumbai','Wankhede Stadium',df['venue'])
df['venue']=np.where(df['venue'] =='MA Chidambaram Stadium, Chepauk, Chennai','MA Chidambaram Stadium',df['venue'])
df['venue']=np.where(df['venue'] =='MA Chidambaram Stadium, Chepauk','MA Chidambaram Stadium',df['venue'])

df['venue']=np.where(df['venue']== 'M.Chinnaswamy Stadium','M Chinnaswamy Stadium',df['venue'])

df['venue']=np.where(df['venue']=='Sardar Patel Stadium, Motera','Narendra Modi Stadium',df['venue'])

df['venue']=np.where(df['venue']=='Feroz Shah Kotla','Arun Jaitley Stadium',df['venue'])


data=df[df['venue'].isin(current_stadium)]

data=data[data['innings']<=2]

index1=data['venue'].value_counts().sort_values(ascending=False).index
values1=data['venue'].value_counts().sort_values(ascending=False).values
d1={ i:j for i,j in zip(index1,values1)}
data['venue']=data['venue'].map(d1)

index2=data['batting_team'].value_counts().sort_values(ascending=False).index
values2=data['batting_team'].value_counts().sort_values(ascending=False).values
d2={ i:j for i,j in zip(index2,values2)}
data['batting_team']=data['batting_team'].map(d2)


index3=data['bowling_team'].value_counts().sort_values(ascending=False).index
values3=data['bowling_team'].value_counts().sort_values(ascending=False).values
d3={ i:j for i,j in zip(index3,values3)}
data['bowling_team']=data['bowling_team'].map(d3)

x=data.drop(['start_date','striker','bowler','total_runs','total_wickets'],axis=1)
y=data['total_runs']

cv=KFold(5)
score=cross_val_score(LinearRegression(),x,y,cv=cv)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=42)

params1={
    'fit_intercept':[True,False],
    'normalize':[True,False]
}

h1=GridSearchCV(LinearRegression(),param_grid=params1,cv=cv)
h1.fit(x_train,y_train)

model1=h1.best_estimator_
model1.fit(x_train,y_train)

pickle.dump(model1,open('hackathon.pkl','wb'))
data.to_csv('useful_data.csv')
# ['Eden Gardens','Wankhede Stadium','M Chinnaswamy Stadium','MA Chidambaram Stadium',
# 'Arun Jaitley Stadium','Narendra Modi Stadium']

# ['Kolkata Knight Riders', 'Royal Challengers Bangalore',
#       'Chennai Super Kings', 'Punjab Kings', 'Rajasthan Royals',
# 'Mumbai Indians','Sunrisers Hyderabad','Delhi Capitals']