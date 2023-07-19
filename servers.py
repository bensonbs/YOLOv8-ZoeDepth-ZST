import io
import os
import re
import cv2
import torch
import base64
import uvicorn
import argparse
import numpy as np
from PIL import Image
from fastapi import FastAPI, Form, UploadFile, File, Body

from ultralytics import YOLO
os.environ['TORCH_HOME'] = '.'

#將base64字串轉換為PIL的image物件
def base64_to_image(base64_str, image_path=None):
    base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
    byte_data = base64.b64decode(base64_data)
    image_data = io.BytesIO(byte_data)
    img = Image.open(image_data).convert("RGB")
    if image_path: 
        img.save(image_path)
    return img

def load_model(Model_name,version):
    if version == 'v5':
        model = torch.hub.load(
            repo_or_dir = 'yolov5',
            model = 'custom', 
            path=f'models/{Model_name}', 
            source='local')
        # 將模型移動到GPU上
    if version == 'v8':
        model = YOLO(f'./models/{Model_name}')

    model.to(DEVICE)
    return model

def outputs_v8(results):
    outputs = {str(i): {
        'name': str(res.names[int(res.boxes.cls)]),
        'class': str(int(res.boxes.cls)),
        'confidence': '{:.2f}'.format(round(float(res.boxes.conf), 2)),
        'xmin': '{:.2f}'.format(res.boxes.xyxy.tolist()[0][0]),
        'ymin': '{:.2f}'.format(res.boxes.xyxy.tolist()[0][1]),
        'xmax': '{:.2f}'.format(res.boxes.xyxy.tolist()[0][2]),
        'ymax': '{:.2f}'.format(res.boxes.xyxy.tolist()[0][3]),
        'keypoints': res.masks.xy[0].tolist()
    } for i, res in enumerate(results[0].cpu()) if round(float(res.boxes.conf), 2) > 0.5}
    return outputs

#設定一個FastAPI的實例
app = FastAPI()
#測試連接
@app.post("/")
def read_root():
    models = os.listdir('models')
    return {model:'available' for model in models}


# 處理上傳圖片的POST請求
@app.post("/upload")
def upload(
    Model_name: str = Body(...),
    base64_str: str = Body(...),
    Deep_model: str = Body(...),
    version: str = Body(...)
    ):
    # try:
    Deep_model = eval(Deep_model)

    if Model_name not in models:
        models[Model_name] = load_model(Model_name,version)
        print(f'>>>>> 讀取 {Model_name} YOLOv5 模型 <<<<<')

    model = models[Model_name]

    # 將base64字串轉換為PIL的image物件
    img_PIL = base64_to_image(base64_str)
    if version == 'v5':
        # 使用YOLOv5模型預測圖片中的物件
        outputs = model(np.array(img_PIL)).pandas().xyxy[0]
        results = {}

        # 如果使用ZoeDepth模型，則需要估算圖片中的物體深度
        if Deep_model:
            # 使用ZoeDepth模型估算深度，並將結果轉換為numpy數組
            depth_numpy = zoe.infer_pil(img_PIL)  
            # 將深度值轉換為介於0到255之間的整數
            depth_int = (255 * (depth_numpy - depth_numpy.min()) / (depth_numpy.max() - depth_numpy.min())).astype(np.uint8)

            # 對於YOLOv5模型預測的每一個物體，都進行深度估算
            for i, co in outputs.iterrows():
                x1,x2 = int(co['xmin']), int(co['xmax'])
                y1,y2 = int(co['ymin']), int(co['ymax'])
                deep = int(np.median(depth_int[y1:y2, x1:x2]))
                results[str(i)] = {c: str(co[c]) for c in co.index}
                results[str(i)]['deep'] = str(deep)
        else:
            # 如果不使用ZoeDepth模型，則直接將YOLOv5模型預測結果轉換為字典格式
            for i, co in outputs.iterrows():
                results[str(i)] = {c: str(co[c]) for c in co.index}

    if version == 'v8':
        outputs = model(np.array(img_PIL))
        results = outputs_v8(outputs)
        if Deep_model:
            # 使用ZoeDepth模型估算深度，並將結果轉換為numpy數組
            depth_numpy = zoe.infer_pil(img_PIL)  
            # 將深度值轉換為介於0到255之間的整數
            depth_int = (255 * (depth_numpy - depth_numpy.min()) / (depth_numpy.max() - depth_numpy.min())).astype(np.uint8)
            for i,co in enumerate(results):
                # Create an empty mask to start with
                mask = np.zeros_like(depth_int, dtype=np.uint8)
                # Fill the area defined by the keypoints in the mask with ones
                cv2.fillConvexPoly(mask, np.array(results[co]['keypoints']).astype(int), 1)
                # Compute the mean depth within the masked area
                mean_depth = np.mean(depth_int[mask == 1])
                results[str(i)]['deep'] = str(mean_depth)
        else:
            for i,co in enumerate(results):
                results[str(i)]['deep'] = str(0)

    # except Exception as e:
    #     return {"message": f"{e}"}

    return results
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0', help='選擇運行設備 (例如: "cuda:0" 或 "cpu")')
    parser.add_argument('--port', type=int, default=5001, help='設定應用程序運行的端口 (例如: "5001" 或 "5002")')
    args = parser.parse_args()

    DEVICE = args.device
    PORT = args.port

    models={}

    print('>>>>> 讀取ZoeDepth距離檢測模型 <<<<<')
    model_zoe_n = torch.hub.load(repo_or_dir = ".", model = "ZoeD_N", source="local", pretrained=True)
        
    print(f'>>>>> 使用 {DEVICE} 進行運算 <<<<<')
    
    zoe = model_zoe_n.to(DEVICE)

    uvicorn.run(app, host='127.0.0.1', port=PORT, log_level="info")
