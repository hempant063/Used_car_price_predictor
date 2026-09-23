from fastapi import FastAPI,Request
from fastapi.templating import Jinja2Templates
from fastapi import Form
import joblib
import pandas as pd
from fastapi.staticfiles import StaticFiles


app=FastAPI()

static=StaticFiles(directory='app/static')
templates=Jinja2Templates(directory='app/templates')

app.mount(
    path="/static",
    app=static,
    name="static",
)
@app.get('/')
def index(request:Request):
    return templates.TemplateResponse(
        request=request,
        name='index.html',
        context={
            'name':'Hans'
        }
    )

@app.get('/prediction')
def predict(request:Request):
    return templates.TemplateResponse(
        request=request,
        name='predict.html',
        
    )


@app.post('/predict')
def prediction(request:Request,
               brand:str=Form(...),
               model:str=Form(...),
               model_year:int=Form(...),
               milage:int=Form(...),
               fuel_type:str=Form(...),
               clean_title:str=Form(...),
               accident:int=Form(...),):
    #predicting the data using MLpipeline
    pipeline=joblib.load('pipeline.pkl')
    df=pd.DataFrame({
        'brand':[brand],
        'model':[model],
        'model_year':[model_year],
        'milage':[milage],
        'fuel_type':[fuel_type],
        'clean_title':[clean_title],
        'accident':[accident]
    })
    prediction=pipeline.predict(df)

    return templates.TemplateResponse(
        request=request,
        name='result.html',
        context={
            'price':round(prediction[0],2)
        }
    )


