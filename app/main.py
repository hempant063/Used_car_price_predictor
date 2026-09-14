from fastapi import FastAPI,Request
from fastapi.templating import Jinja2Templates
from fastapi import Form

app=FastAPI()

templates=Jinja2Templates(directory='app/templates')

@app.get('/')
def index(request:Request):
    return templates.TemplateResponse(
        request=request,
        name='index.html',
        context={
            'name':'Hans'
        }
    )

@app.post('/predict')
def prediction(request:Request,
               company:str=Form(...),
               age:int=Form(...)):
    return templates.TemplateResponse(
        request=request,
        name='index.html',
        context={
            'company':company,
            'age':age
        }
    )

