# YOLOv8-ZoeDepth-ZST

## 簡介
此API使用了基於Pytorch的YOLO模型進行影像中的物件偵測。
我們支援兩種版本的YOLO，
- `YOLOv5`
- `YOLOv8`

此外，還可以選擇使用`ZoeDepth`模型進行物件深度的估算。

## API端點
以下是此API的主要端點:
- /: 顯示可用模型列表的端點
- /upload: 上傳圖片並進行物件偵測的端點

### 端點 /
**此端點用於取得當前可用的YOLO模型列表。**

方法: POST
回傳格式: JSON
回傳內容: 一個字典，鍵為模型名稱，值為'available'

## 端點 /upload
**此端點接受上傳的圖片並進行物件偵測。**

方法: POST
參數:
- Model_name: 選擇YOLO模型名稱，需符合當前可用模型列表的模型名稱。
- base64_str: 圖片的base64編碼字串，需包含 data:image/[image format];base64, 的前綴，例如 data:image/jpeg;base64,/9j/4AAQSk...
- Deep_model: 選擇是否使用ZoeDepth模型進行深度估算，需傳入Python的bool值字串，例如 "True" 或 "False"。
- version: YOLO模型版本，目前支援 "v5" 或 "v8"。

回傳格式: JSON
- 回傳內容: 每一個偵測到的物件都會有一個字典，字典的鍵包含 `name`、`class`、`confidence`、`xmin`、`ymin`、`xmax`、`ymax`
- 使用深度估算時會有 `deep`。
- 如果使用的是YOLOv8，還會包含 `keypoints`。

## Demo.ipynb
**使用Yolov5偵測物體bbox後，將擷取bbox內深度資訊中位數，顯示於DS**
```
version = 'v5'
Model_name = 'TC3-PPE_Detector_v3.pt'
Deep_model = True

r = requests.post('http://127.0.0.1:5001/upload', 
json = {
    'base64_str':f'{img_base64}',
    'Model_name': Model_name,
    'Deep_model':Deep_model,
    'version':'v5'
})

dic = json.loads(r.text)
dicplot(image,dic,version)
```
![Optional alt text](./yolov5-ZoeDepth_ZST.png)

**使用Yolov8-seg偵測物體keypoint後，將擷取mask內深度資訊平均數，顯示於DS**
```
version = 'v8'
Model_name = 'best.pt'
Deep_model = True

r = requests.post('http://127.0.0.1:5001/upload', 
json = {
    'base64_str':f'{img_base64}',
    'Model_name': Model_name,
    'Deep_model':Deep_model,
    'version':version
})

dic = json.loads(r.text)
dicplot(image,dic,version)
```
![Optional alt text](./yolov8-ZoeDepth_ZST.png)
