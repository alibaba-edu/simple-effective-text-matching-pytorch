from fastapi import FastAPI
from test import Testor
import uvicorn

api = FastAPI()


@api.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")




@api.post("/check_text_match/")
async def check_text_match(Text1: str, Text2: str):
    model_path='./models/snli/benchmark-0/best.pt'
    data=[Text1, Text2]
    Test=Testor(model_path,data)
     
    return Test.Run()




