import streamlit as st
import os
import requests
import base64
import io
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from roboflow import Roboflow


## store initial session state values
project_url_od, private_api_key, uploaded_file_od = ("", "", "")

if 'project_url_od' not in st.session_state:
    st.session_state['project_url_od'] = ""
if 'confidence_threshold' not in st.session_state:
    st.session_state['confidence_threshold'] = "40"
if 'overlap_threshold' not in st.session_state:
    st.session_state['overlap_threshold'] = "30"
if 'include_bbox' not in st.session_state:
    st.session_state['include_bbox'] = "Yes"
if 'show_class_label' not in st.session_state:
    st.session_state['show_class_label'] = 'Show Labels'
if 'box_type' not in st.session_state:
    st.session_state['box_type'] = "regular"
if 'private_api_key' not in st.session_state:
    st.session_state['private_api_key'] = ""
if 'uploaded_file_od' not in st.session_state:
    st.session_state['uploaded_file_od'] = ""

##########
#### Set up main app logic
##########
def run_inference(workspace_id, model_id, version_number, uploaded_img, inferenced_img):
    rf = Roboflow(api_key=st.session_state['private_api_key'])
    project = rf.workspace(workspace_id).project(model_id)
    project_metadata = project.get_version_information()
    version = project.version(version_number)
    model = version.model

    if project.type != "object-detection":
        st.write("### Please include the project URL for an object detection model trained with Roboflow Train")

    project_type = st.write(f"#### Project Type: {project.type}")
    for i in range(len(project_metadata)):
        if project_metadata[i]['id'] == extracted_url:
            project_endpoint = st.write(f"#### Inference Endpoint: {project_metadata[i]['model']['endpoint']}")
            st.write(f"#### Model ID: {model_id}")
            st.write(f"#### Version Name: {project_metadata[i]['name']}")
            st.write(f"Input Image Size for Model Training (pixels, px):")
            width_metric, height_metric = st.columns(2)
            width_metric.metric(label='Pixel Width', value=project_metadata[i]['preprocessing']['resize']['width'])
            height_metric.metric(label='Pixel Height', value=project_metadata[i]['preprocessing']['resize']['height'])

            if project_metadata[i]['model']['fromScratch']:
                train_checkpoint = 'Scratch'
                st.write(f"#### Version trained from {train_checkpoint}")
            elif project_metadata[i]['model']['fromScratch'] is False:
                train_checkpoint = 'Checkpoint'
                st.write(f"#### Version trained from {train_checkpoint}")
            else:
                train_checkpoint = 'Not Yet Trained'
                st.write(f"#### Version is {train_checkpoint}")

    st.write("#### Uploaded Image")
    st.image(uploaded_img, caption="Uploaded Image", use_column_width=True)

    predictions = model.predict(uploaded_img, overlap=int(st.session_state['overlap_threshold']),
        confidence=int(confidence_threshold), stroke=int(st.session_state['box_width']))

    predictions_json = predictions.json()

    # drawing bounding boxes with the Pillow library
    # https://docs.roboflow.com/inference/hosted-api#response-object-format
    collected_predictions = []
    for bounding_box in predictions:
        x0 = bounding_box['x'] - bounding_box['width'] / 2
        x1 = bounding_box['x'] + bounding_box['width'] / 2
        y0 = bounding_box['y'] - bounding_box['height'] / 2
        y1 = bounding_box['y'] + bounding_box['height'] / 2
        class_name = bounding_box['class']
        confidence_score = bounding_box['confidence']
        box = (x0, x1, y0, y1)
        detected_x = int(bounding_box['x'] - bounding_box['width'] / 2)
        detected_y = int(bounding_box['y'] - bounding_box['height'] / 2)
        detected_width = int(bounding_box['width'])
        detected_height = int(bounding_box['height'])
        # ROI (Region of Interest), or detected bounding box area
        roi_bbox = [detected_y, detected_height, detected_x, detected_width]
        collected_predictions.append({"class":class_name, "confidence":confidence_score,
                                    "x0,x1,y0,y1":[int(x0),int(x1),int(y0),int(y1)],
                                    "Width":int(bounding_box['width']), "Height":int(bounding_box['height']),
                                    "ROI, bbox (y+h,x+w)":roi_bbox,
                                    "bbox area (px)":abs(int(x0)-int(x1))*abs(int(y0)-int(y1))})
        # position coordinates: start = (x0, y0), end = (x1, y1)
        # color = RGB-value for bounding box color, (0,0,0) is "black"
        # thickness = stroke width/thickness of bounding box
        start_point = (int(x0), int(y0))
        end_point = (int(x1), int(y1))
        if show_box_type == 'regular':
            if show_bbox == 'Yes':
                # draw/place bounding boxes on image
                cv2.rectangle(inferenced_img, start_point, end_point, color=(0,0,0), thickness=int(st.session_state['box_width']))

            if show_class_label == 'Show Labels':
                # add class name with filled background
                cv2.rectangle(inferenced_img, (int(x0), int(y0)), (int(x0) + 40, int(y0) - 20), color=(0,0,0),
                        thickness=-1)
                cv2.putText(inferenced_img,
                    class_name,#text to place on image
                    (int(x0), int(y0) - 5),#location of text
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,#font
                    fontScale=0.4,#font scale
                    color=(255,255,255),#text color
                    thickness=int(st.session_state['text_width'])#thickness/"weight" of text
                    )

        if show_box_type == 'fill':
            if show_bbox == 'Yes':
                # draw/place bounding boxes on image
                cv2.rectangle(inferenced_img, start_point, end_point, color=(0,0,0), thickness=-1)

            if show_class_label == 'Show Labels':
                # add class name with filled background
                cv2.rectangle(inferenced_img, (int(x0), int(y0)), (int(x0) + 40, int(y0) - 20), color=(0,0,0),
                        thickness=-1)
                cv2.putText(inferenced_img,
                    class_name,#text to place on image
                    (int(x0), int(y0) - 5),#location of text
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,#font
                    fontScale=0.3,#font scale
                    color=(255,255,255),#text color
                    thickness=int(st.session_state['text_width'])#thickness/"weight" of text
                    )

        if show_box_type == 'blur':
            if show_bbox == 'Yes':
                # draw/place bounding boxes on image
                cv2.rectangle(inferenced_img, start_point, end_point, color=(0,0,0), thickness=int(st.session_state['box_width']))
            
            box = [(x0, y0), (x1, y1)]
            blur_x = int(bounding_box['x'] - bounding_box['width'] / 2)
            blur_y = int(bounding_box['y'] - bounding_box['height'] / 2)
            blur_width = int(bounding_box['width'])
            blur_height = int(bounding_box['height'])
            # ROI (Region of Interest), or area to blur
            roi = inferenced_img[blur_y:blur_y+blur_height, blur_x:blur_x+blur_width]

            # ADD BLURRED BBOXES
            # set blur to (31,31) or (51,51) based on amount of blur desired
            if st.session_state["amount_blur"] == "High":
                blur_image = cv2.GaussianBlur(roi,(51,51),0)
            elif st.session_state["amount_blur"] == "Low":
                blur_image = cv2.GaussianBlur(roi,(31,31),0)
            inferenced_img[blur_y:blur_y+blur_height, blur_x:blur_x+blur_width] = blur_image

            if show_class_label == 'Show Labels':
                # add class name with filled background
                cv2.rectangle(inferenced_img, (int(x0), int(y0)), (int(x0) + 40, int(y0) - 20), color=(0,0,0),
                        thickness=-1)
                cv2.putText(inferenced_img,
                    class_name,#text to place on image
                    (int(x0), int(y0) - 5),#location of text
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,#font
                    fontScale=0.3,#font scale
                    color=(255,255,255),#text color
                    thickness=int(st.session_state['text_width'])#thickness/"weight" of text
                    )

    ## Subtitle.
    st.write("### Inferenced Image")    
    st.image(inferenced_img, caption="Inferenced Image", use_column_width=True)

    results_tab, json_tab, project_tab = st.tabs(["Inference Results", "JSON Response Output", "Project Info"])

    with results_tab:
        ## Display results dataframe in main app.
        st.write('### Prediction Results (Pandas DataFrame)')
        st.dataframe(collected_predictions)

    with json_tab:
        ## Display the JSON in main app.
        st.write('### JSON Output')
        st.write(predictions_json)

    with project_tab:
        st.write(f"Annotation Group Name: {project.annotation}")
        col1, col2, col3 = st.columns(3)
        col1.write(f'Total images in the version: {version.images}')
        col1.metric(label='Augmented Train Set Image Count', value=version.splits['train'])
        for i in range(len(project_metadata)):
            if project_metadata[i]['id'] == extracted_url:
                col2.metric(label='mean Average Precision (mAP)', value=f"{float(project_metadata[i]['model']['map'])}%")
                col2.metric(label='Precision', value=f"{float(project_metadata[i]['model']['precision'])}%")
                col2.metric(label='Recall', value=f"{float(project_metadata[i]['model']['recall'])}%")
        
        col3.metric(label='Train Set Image Count', value=project.splits['train'])
        col3.metric(label='Valid Set Image Count', value=project.splits['valid'])
        col3.metric(label='Test Set Image Count', value=project.splits['test'])

        col4, col5, col6 = st.columns(3)
        col4.write('Preprocessing steps applied:')
        col4.json(version.preprocessing)
        col5.write('Augmentation steps applied:')
        col5.json(version.augmentation)
        col6.metric(label='Train Set', value=version.splits['train'], delta=f"Increased by Factor of {(version.splits['train'] / project.splits['train'])}")
        col6.metric(label='Valid Set', value=version.splits['valid'], delta="No Change")
        col6.metric(label='Test Set', value=version.splits['test'], delta="No Change")

    st.write("### Code to Generate the Image and Inference Results:")
    st.write("##### https://docs.roboflow.com/python#finding-your-project-information-manually")

    inference_code = f'''rf = Roboflow(api_key=ROBOFLOW_API_KEY)
workspace = rf.workspace({workspace_id})
project = workspace.project({model_id})
version = project.version({version_number})
model = version.model

img = cv2.imread(img_path)
## perform inference on the selected image
predictions = model.predict(img_path, confidence={st.session_state["confidence_threshold"]},
    overlap={st.session_state["overlap_threshold"]})'''

    customization_code = ""

    if st.session_state["box_type"] == "regular":
        customization_code = f"""## https://docs.roboflow.com/inference/hosted-api#response-object-format
for bounding_box in predictions:
    ## defining bounding box area and saving class name and confidence scores
    x0 = bounding_box['x'] - bounding_box['width'] / 2#start_column
    x1 = bounding_box['x'] + bounding_box['width'] / 2#end_column
    y0 = bounding_box['y'] - bounding_box['height'] / 2#start row
    y1 = bounding_box['y'] + bounding_box['height'] / 2#end_row
    start_point = (int(x0), int(y0))# bounding box start point (pt1)
    end_point = (int(x1), int(y1))# bounding box end point (pt2)

    ## draw/place bounding boxes on image
    cv2.rectangle(img, start_point, end_point, color=(0,0,0), thickness={st.session_state['box_width']})
    
cv2.imshow('regular bounding boxes', img)"""

    if st.session_state["box_type"] == "fill":
        customization_code = """## https://docs.roboflow.com/inference/hosted-api#response-object-format
for bounding_box in predictions:
    ## defining bounding box area and saving class name and confidence scores
    x0 = bounding_box['x'] - bounding_box['width'] / 2#start_column
    x1 = bounding_box['x'] + bounding_box['width'] / 2#end_column
    y0 = bounding_box['y'] - bounding_box['height'] / 2#start row
    y1 = bounding_box['y'] + bounding_box['height'] / 2#end_row
    start_point = (int(x0), int(y0))# bounding box start point (pt1)
    end_point = (int(x1), int(y1))# bounding box end point (pt2)

    ## draw/place bounding boxes on image
    cv2.rectangle(img, start_point, end_point, color=(0,0,0), thickness=-1)
    
cv2.imshow('filled bounding boxes', img)"""

    if st.session_state["box_type"] == "blur":
        blur_value = 0
        if st.session_state["amount_blur"] == "High":
            blur_value = 51
        elif st.session_state["amount_blur"] == "Low":
            blur_value = 31
        customization_code = f"""## https://docs.roboflow.com/inference/hosted-api#response-object-format
for bounding_box in predictions:
    ## defining bounding box area to blur
    blur_x = int(bounding_box['x'] - bounding_box['width'] / 2)
    blur_y = int(bounding_box['y'] - bounding_box['height'] / 2)
    blur_width = int(bounding_box['width'])
    blur_height = int(bounding_box['height'])
    ## region of interest (ROI), or area to blur
    roi = img[blur_y:blur_y+blur_height, blur_x:blur_x+blur_width]

    ## ADD BLURRED BBOXES
    ## set blur to (31,31) or (51,51) based on amount of blur desired
    blur_image = cv2.GaussianBlur(roi,({blur_value},{blur_value}),0)
    img[blur_y:blur_y+blur_height, blur_x:blur_x+blur_width] = blur_image

cv2.imshow('blurred bounding boxes', img)"""

    st.code(inference_code, language='python')
    st.code(customization_code, language='python')


##########
##### Set up sidebar.
##########
# Add in location to select image.
with st.sidebar:
    st.write("#### Select an image to upload.")
    uploaded_file_od = st.file_uploader("Image File Upload",
                                        type=["png", "jpg", "jpeg"],
                                        accept_multiple_files=False)

    st.write("[Find additional images on Roboflow Universe.](https://universe.roboflow.com/)")
    st.write("[Improving Your Model with Active Learning](https://help.roboflow.com/implementing-active-learning)")

    ## Add in sliders.
    confidence_threshold = st.slider("Confidence threshold (%): What is the minimum acceptable confidence level for displaying a bounding box?", 0, 100, 40, 1)
    overlap_threshold = st.slider("Overlap threshold (%): What is the maximum amount of overlap permitted between visible bounding boxes?", 0, 100, 30, 1)

    col_bbox, col_blur, col_labels = st.columns(3)
    
    with col_bbox:
        show_bbox = st.radio("Show Bounding Boxes:",
                            options=["Yes", "No"],
                            index=0,
                            key="include_bbox")

    with col_blur:
        amount_blur = st.radio("Amount of Blur:",
                                options=["Low", "High"],
                                index=1,
                                key="amount_blur")

    with col_labels:
        show_class_label = st.radio("Show Class Labels:",
                                    options=["Show Labels", "Hide Labels"],
                                    index=0,
                                    key="show_class_label")

    show_box_type = st.selectbox("Display Bounding Boxes As:",
                                options=("regular", "fill", "blur"),
                                index=0,
                                key="box_type")

    box_width = st.selectbox("Width of Bounding Boxes:",
                            options=("1", "2", "3", "4", "5"),
                            index=1,
                            key="box_width")
    
    text_width = st.selectbox("Thickness of Label Text:",
                            options=("1", "2", "3"),
                            index=0,
                            key="text_width")
        
##########
##### Set up project access.
##########

## Title.
st.write("# Roboflow Object Detection Tests")

with st.form("project_access"):
    st.write("#### Select 'Verify and Load Model' after entering Project URL and API Key to receive predictions")
    st.write("No trained models, or Roboflow projects? Use one of over 150,000 datasets and 25,000 models on [Roboflow Universe](https://universe.roboflow.com/), free!")
    st.write("Using Roboflow Universe: https://blog.roboflow.com/computer-vision-datasets-and-apis/")
    project_url_od = st.text_input("Project URL", key="project_url_od",
                                help="Copy/Paste Your Project URL: https://docs.roboflow.com/python#finding-your-project-information-manually",
                                placeholder="https://app.roboflow.com/workspace-id/model-id/version")
    private_api_key = st.text_input("Private API Key", key="private_api_key", type="password",placeholder="Input Private API Key")
    submitted = st.form_submit_button("Verify and Load Model")
    st.write("*** Don't forget to upload an image (jpg, jpeg, or png) to be predicted!!! ***")
    if submitted:
        st.write("Loading model...")
        extracted_url = project_url_od.split("roboflow.com/")[1]
        if "model" in project_url_od.split("roboflow.com/")[1]:
            workspace_id = extracted_url.split("/")[0]
            model_id = extracted_url.split("/")[1]
            version_number = extracted_url.split("/")[3]
        elif "deploy" in project_url_od.split("roboflow.com/")[1]:
            workspace_id = extracted_url.split("/")[0]
            model_id = extracted_url.split("/")[1]
            version_number = extracted_url.split("/")[3]
        else:
            workspace_id = extracted_url.split("/")[0]
            model_id = extracted_url.split("/")[1]
            version_number = extracted_url.split("/")[2]

if uploaded_file_od != None:
    # User-selected image.
    image = Image.open(uploaded_file_od)
    uploaded_img = np.array(image)
    inferenced_img = uploaded_img.copy()

    run_inference(workspace_id, model_id, version_number, uploaded_img, inferenced_img)
