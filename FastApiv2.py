from distutils.command.install import INSTALL_SCHEMES
from email import message
from mailbox import Message
from xml.dom import INVALID_CHARACTER_ERR
from fastapi import FastAPI,Form,File,UploadFile
from fastapi.responses import JSONResponse,HTMLResponse
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import time
from yolov7.detectCostum import detect
import cv2
import sys
from skimage.exposure import is_low_contrast
import imageio.v2 as imageio
import numpy as np
from fastapi import Security,Depends,HTTPException
from fastapi.security.api_key import APIKeyQuery,APIKey
from starlette.status import HTTP_403_FORBIDDEN
from io import StringIO
import urllib
import base64
import requests
import os, shutil
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates




weight = "C:/Python/FastApi/FastApi/best.pt"

app = FastAPI(
    docs_url="/docs",
    title = "The Coolest Object Detector API",
    description= "This API provides object detection handling and gives you back the found people in an image",
    version="1.0.0"
    )

templates = Jinja2Templates(directory="templates")
app.mount("/static",StaticFiles(directory="static"),name="static")

class BasicResponse(BaseModel):
    status_code: int
    message: str
    content: Optional[dict]

@app.get("/", response_model=BasicResponse)

def root():
    return BasicResponse(status_code=200,
                         message="healthy", content={
                             "timestamp": datetime.now()
                             })

class DetectionInput(BaseModel):
    file: UploadFile

    @classmethod
    def as_form(cls,file:UploadFile = File(...)):
        return cls(file=File)





TOKEN = "coolesttoken"
api_key_query = APIKeyQuery(name="token",auto_error=False)


async def get_api_key(api_key_query: str = Security(api_key_query)):
    if api_key_query == TOKEN:
        return api_key_query
    else:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,detail="Could not validate credentials"
            )



@app.post("/detect",
            response_model=BasicResponse
            )

def detection(inp: DetectionInput,api_key:APIKey = Depends(get_api_key)):
    time_start = time.time()


    im = bytes(inp.file, "utf-8").decode("unicode_escape")


    #Dark
    img = cv2.imread("image.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    

    if(is_low_contrast(gray,0.35)):
        time_end = round(time.time()-time_start,2)
        cv2.putText(img, "low contrast image", (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0,0,0), 2)
        return BasicResponse(status_code=200,
                                message="success",
                                content={
                                    "label": "Image is dark",
                                    "computation_time": f"{time_end} seconds",
                                    "timestamp": datetime.now()

                                    })
        


    else:

        #Blur check
        img2 = cv2.imread("image.jpg",cv2.IMREAD_GRAYSCALE)
        laplacian_var = cv2.Laplacian(img2, cv2.CV_64F).var()


        if laplacian_var <=160:
            time_end = round(time.time()-time_start,2)
            return BasicResponse(status_code=200,
                                message="success",
                                content={
                                    "label": "Image is blur",
                                    "computation_time": f"{time_end} seconds",
                                    "timestamp": datetime.now()

                                    })

        else:

            try:
                label = detect(weights=weight,source="image.jpg",conf_thres=0.5)


            except Exception as e:

                return BasicResponse(status_code=400,
                                        message=str(e),content={"timestamp":datetime.now()
                                            }
                    )
            time_end = round(time.time()-time_start,2)





            return BasicResponse(status_code=200,
                                    message="success",
                                    content={
                                        "label": label,
                                        "computation_time": f"{time_end} seconds",
                                        "timestamp": datetime.now()

                                        })
    os.remove("image.jpg")                                                    
    
        
