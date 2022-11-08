from distutils.command.install import INSTALL_SCHEMES
from email import message
from mailbox import Message
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import time
from yolov7onnxruntime.yolov7 import yolov7


yolov7_detector = yolov7("C:/Python/FastApi/FastApi/yolov7onnxruntime/models/yolov7-tiny_640x640.onnx",conf_thres=0.5,iou_thres=0.5)

app = FastAPI(
    docs_url="/docs",
    title = "The Coolest Object Detector API",
    description= "This API provides object detection handling and gives you back the found people in an image",
    version="1.0.0"
    )

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
    image: str


    class Config:
        schema_extra = {
            "example" : {
                "image":"https://raw.githubusercontent.com/ucekmez/yolov7onnxruntime/main/sample.jpg"
                        }                      
                       }


@app.post("/detect",
          response_model=BasicResponse
          )

def detection(inp: DetectionInput):
    time_start = time.time()

    try:
        boxes,scores,class_ids = yolov7_detector(inp.image)

    except Exception as e:

        return BasicResponse(status_code=400,
                             message=str(e),content={"timestamp":datetime.now()
                                 }
            )
    time_end = round(time.time()-time_start,2)

    class_ids_to_names = [yolov7_detector.class_names[i] for i in class_ids]

    return BasicResponse(status_code=200,
                         message="success",
                         content={
                             "boxes":boxes.tolist(),
                             "scores": class_ids_to_names,
                             "computation_time": f"{time_end} seconds",
                             "timestamp": datetime.now()

                             })
        
