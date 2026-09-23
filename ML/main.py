# import pandas as pd
# import streamlit as st
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression,Ridge,Lasso
# from sklearn.preprocessing import StandardScaler,OneHotEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
# import joblib
# from sklearn.metrics import mean_absolute_error as me,mean_squared_error as mse,r2_score



# if __name__=="__main__":

#     #Loading the dataset 
#     try:
#         df=joblib.load('pipeline.pkl')
#         st.success('Pipeline loaded')
#     except:
#         try:
#             df=pd.read_csv('dataset/used_cars.csv')
#             st.success('File loaded for Machine Learning')
#         except:
#             st.error('File not found')


#     #Cleaning the dataset for ML model

#     df['price']=df['price'].str.replace("$",'').str.replace(",",'').astype(int)
#     df['milage']=df['milage'].str.replace('mi.','').str.replace(',','').astype(int)
#     null_values=['fuel_type','accident','clean_title']
#     df['clean_title']=df['clean_title'].fillna('No')
#     df['fuel_type']=df['fuel_type'].fillna('Electricity')

#     #preprocessing
#     numerical_features=['model_year','milage','accident']
#     categorical_features=['brand','model','fuel_type','clean_title']

#     numerical_transformer=Pipeline([
#         ('imputer',SimpleImputer(strategy='median')),
#         ('scalar',StandardScaler()),
#     ])

#     categorical_transformer=Pipeline([
#         ('imputer',SimpleImputer(strategy='constant',fill_value='Missing')),
#         ('encoder',OneHotEncoder(handle_unknown='ignore'))
#     ])

#     preprocessor=ColumnTransformer([
#         ('numerical',numerical_transformer,numerical_features),
#         ('categorical',categorical_transformer,categorical_features)
#     ])

#     pipeline=Pipeline([
#         ('preprocessing',preprocessor),
#         ('model',Lasso())
#     ])

#     #Preparing training data to feed to the pipeline 

#     X=df.drop(columns=['price'])
#     y=df['price']

#     X_train,X_test,y_train,y_test=train_test_split(
#         X,
#         y,
#         test_size=0.2,
#         random_state=42
#     )


#     cv_score=cross_val_score(
#         pipeline,
#         X_train,
#         y_train,
#         cv=5,
#         scoring='r2'
#     )

#     st.write('Mean Cross Validation Score: ',cv_score.mean())
#     pipeline.fit(X_train,y_train)
#     predictions=pipeline.predict(X_test)

#     #Drawing the graph

#     minv=min(predictions.min(),y_test.min())
#     maxv=max(predictions.max(),y_test.max())

#     fig,ax=plt.subplots()
#     ax.scatter(predictions,y_test)
#     ax.plot(
#         [minv,maxv],
#         [minv,maxv],
#         'g'
#     )
#     st.pyplot(fig)

#     #Drawing residual plot

#     fig2,ax2=plt.subplots()
#     ax2.scatter(y_test,y_test-predictions)
#     ax2.plot(
#         [0,maxv],
#         [0,0],
#         'b'
#     )
#     st.pyplot(fig2)

#     st.write('Mean Abosulute Error: ',me(predictions,y_test))
#     st.write('Mean Squared Error: ',mse(predictions,y_test))
#     st.write('R2 score: ',r2_score(predictions,y_test))



# Starting Once Again

import numpy as np,pandas as pd,matplotlib.pyplot as plt, streamlit as st
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.metrics import r2_score,mean_absolute_error as me, mean_squared_error as mse


if __name__=="__main__":

    #Loading the csv file into the program
    is_pkl_not_there=True
    try:
        df=joblib.load('pipeline.pkl')
        st.success('Pipeline loaded')
        is_pkl_not_there=False
    except:
        pass
        

    if is_pkl_not_there:
        try:
            df=pd.read_csv('dataset/used_cars.csv')
            # st.success('File loaded successfully')
        except:
            print("File not found")
            # st.error('File not found')
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
            ('model',Ridge(0.1))
        ])

        X=df.drop(columns='price')
        y=df['price']

        # X_train,X_test,y_train,y_test=train_test_split(
        #     X,
        #     y,
        #     test_size=0.2,
        #     random_state=42
        # )

        # cv_score=cross_val_score(
        #     pipeline,
        #     X_train,
        #     y_train,
        #     cv=5,
        #     scoring='r2'
        # )
        # st.write(cv_score)

        pipeline.fit(X,y)
        # predictions=pipeline.predict(X_test)
        # st.write('Train R2 score: ',pipeline.score(X_train,y_train))
        # st.write('Test R2 score: ',pipeline.score(X_test,y_test))
        # st.write('Test ME score: ',me(y_test,predictions))
        # st.write('Test MSE score: ',mse(y_test,predictions))

        joblib.dump(pipeline,'pipeline.pkl')

    #Drawing the graphs for evaluation
    # minv=min(y_test.min(),predictions.min())
    # maxv=max(y_test.max(),predictions.max())

    # fig,ax=plt.subplots()
    # ax.scatter(y_test,predictions)
    # ax.plot(
    #     [minv,maxv],
    #     [minv,maxv],
    #     'b'
    # )
    # st.pyplot(fig)

    # #Drawing residual plot
    # residuals=y_test-predictions
    # fig2,ax2=plt.subplots()
    # ax2.scatter(y_test,residuals)
    # ax2.plot(
    #     [0,maxv],
    #     [0,0],
    #     'g'
    # )
    # st.pyplot(fig2)

    # #use grid seachcv to estimate the good value of alpha in ridge as well as lasso
    # param_grid={
    #     'model__alpha':[0.001,0.01,0.1,1]
    # }
    # grid=GridSearchCV(
    #     pipeline,
    #     param_grid,
    #     cv=5,
    #     scoring='r2'
    # )
    # grid.fit(X_train,y_train)
    # st.write('best score is: ',grid.best_score_)
    # st.write('best estimator is: ',grid.best_params_)

    #Ridge(0.1) was the best scorer with a score of 0.7335