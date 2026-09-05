from fastapi import FastAPI
import uvicorn


app=FastAPI()

@app.get('/')
def greet():
    return {'Hello world'}