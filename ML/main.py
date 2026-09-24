
#0.66 r2_score

import numpy as np,pandas as pd,matplotlib.pyplot as plt, streamlit as st
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.metrics import r2_score,mean_absolute_error as me, mean_squared_error as mse


if __name__=="__main__":

    #Loading the csv file into the program
    is_pkl_not_there=True
    # try:
    #     df=joblib.load('pipeline.pkl')
    #     st.success('Pipeline loaded')
    #     is_pkl_not_there=False
    # except:
    #     pass
        

    if is_pkl_not_there:
        try:
            @st.cache_data
            def loadcsv():
                df=pd.read_csv('dataset/used_cars.csv')
                return df
            df=loadcsv()
            st.success('File loaded successfully')
        except:
            st.error('File not found')
        #Cleaning the data using pandas to feed to the model
        df['milage']=df['milage'].str.replace(',','').str.replace('mi.','').astype(int)
        df['price']=df['price'].str.replace('$','').str.replace(',','').astype(int)
        df['fuel_type']=df['fuel_type'].fillna('Electricity')
        df['clean_title']=df['clean_title'].fillna('No')
        # st.write(df.head(10))

        #preprocessing the data to feed to the ML model
        numerical_features=['model_year','milage','accident'] 
        categorical_features=['brand','model','fuel_type','clean_title']

        #making transformers
        numerical_transformer=Pipeline([
            ('imputer',SimpleImputer(strategy='median')),
            ('scalar',StandardScaler())
        ])

        categorical_transformer=Pipeline([
            ('imputer',SimpleImputer(strategy='constant',fill_value='missing',add_indicator=False)),
            ('encoder',OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor=ColumnTransformer([
            ('numerical',numerical_transformer,numerical_features),
            ('categorical',categorical_transformer,categorical_features)
        ])

        pipeline=Pipeline([
            ('preprocessor',preprocessor),
            ('model',GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.06
            ))
        ])

        X=df.drop(columns='price')
        y=df['price']

        X_train,X_test,y_train,y_test=train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        pipeline.fit(X_train,y_train)

        predictions=pipeline.predict(X_test)
        # st.write('Train R2 score: ',r2_score(y_train,train_predictions))
        # st.write('Test R2 score: ',r2_score(y_test,predictions))
        # st.write('Test ME score: ',me(y_test,predictions))
        # st.write('Test MSE score: ',mse(y_test,predictions))

        joblib.dump(pipeline,'pipeline.pkl')

    # Drawing the graphs for evaluation
    # minv=min(y_test.min(),predictions.min())
    # maxv=max(y_test.max(),predictions.max())

    # fig,ax=plt.subplots()
    # ax.plot(
    #     [minv,maxv],
    #     [minv,maxv],
    #     'b',
    #     label='best-fit line'
    # )
    # residuals=y_test-predictions
    # ax.scatter(y_test,residuals,label='y_test vs residuals')
    # ax.scatter(y_test,predictions,color='orange',label='y_test vs predictions')
    # ax.plot(
    #     [0,maxv],
    #     [0,0],
    #     'red',
    #     label='zero-error line'
    # )
    # ax.legend()
    # st.pyplot(fig)

    # #Drawing residual plot
    # fig2,ax2=plt.subplots()
    # st.pyplot(fig2)


    # #Ridge(0.1) was the best scorer with a score of 0.7335