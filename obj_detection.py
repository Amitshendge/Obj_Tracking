from dotenv import load_dotenv
import os
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np

predictionENDPOINT = os.getenv("predictionENDPOINT")
predictionKey = os.getenv("predictionKey")
ProjectID=os.getenv("ProjectID")
ModelName=os.getenv("ModelName")


class Obj_detect:
    def __init__(self) -> None:
        #Authenticate client for training API
        self.credentials=ApiKeyCredentials(in_headers={"Prediction-key":predictionKey})
        self.prediction_client=CustomVisionPredictionClient(endpoint=predictionENDPOINT,credentials=self.credentials)
    def main(self):
        #Load image and get height, width and channels
        image_file="frame.jpg"
        print("Detecting Objects in image")
        image=Image.open(image_file)
        h,w,ch=np.array(image).shape

        #Detect objects in the test image
        with open(image_file,mode="rb") as image_data:
            results = self.prediction_client.detect_image(ProjectID,ModelName,image_data)

        tag_name_lst=[]
        probability_lst=[]
        box_lst=[]
        for prediction in results.predictions:
            #Only show objects with 50% probability or accuracy
            if(((prediction.probability*100)>70) & ((prediction.tag_name) != "Worker")):
                #Box co-ordinates and dimentions are  proportional
                left=int(prediction.bounding_box.left * w)
                top=int(prediction.bounding_box.top * h)
                height=int(prediction.bounding_box.height * h)
                width=int(prediction.bounding_box.width * w)
                #Draw the box
                # points=((left,top),(left+width,top),(left+width,top+height),(left,top+height),(left,top))
                #Add the tag name and probability
                tag_name_lst.append(prediction.tag_name)
                probability_lst.append(prediction.probability)
                box_lst.append((left,top,height,width))

        return (tag_name_lst,probability_lst,box_lst)


if __name__=="__main__":
    A=Obj_detect()
    print(A.main())


















