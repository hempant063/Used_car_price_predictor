from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi import Form


templates=Jinja2Templates(directory='app/templates')

app=FastAPI()

@app.get('/')
def home(request:Request):
    return templates.TemplateResponse(
        request=request,
        name='index.html',
        context={
            'page':'Home'
        }
    )


@app.post('/predict')
def prediction(request:Request,company:str=Form(...),age:int=Form(...)):
    return templates.TemplateResponse(
        request=request,
        name='index.html',
        context={
            'company':company,
            'age':age,
        }
    )