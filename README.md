# MRI Tumor Detection using yolov5
* Clone the ultralytics/yolov5 repository from Github using the following link: 
```
git clone https://github.com/ultralytics/yolov5.git
```
* Run the following command to install the required libraries
```
pip install -r yolov5/requirements.txt
```
* Download the dataset from kaggle using this command
```
kaggle datasets download -d ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes
```

* Use model_training.py for preprocessing and creating .yaml file
* After this, use the following command to start the training: 
```
python train.py --img 640 --batch 16 --epochs 15 --data path/to/your/dataset.yaml --weights yolov5s.pt
```


* You can use model_export.py to copy the model to parent directory.
* Using streamlit, you can deploy the app.py using your local machine or streamlit community cloud
