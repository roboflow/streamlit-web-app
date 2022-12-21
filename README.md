<h1 align="center">Roboflow: Streamlit Computer Vision App for Web Browser Deployment</h1>
<h2 align="center">A web-based application for testing models trained with Roboflow. Powered by Streamlit.</h2>

<br></br>
[![Roboflow](https://raw.githubusercontent.com/roboflow-ai/notebooks/main/assets/badges/roboflow-blogpost.svg)](https://blog.roboflow.com/)
[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/@Roboflow)

## 🎬 videos

Every week we create tutorials showing you the best new updates in Computer Vision. 🔥
[Subscribe](https://www.youtube.com/@Roboflow), and stay up to date with our latest YouTube videos!

## How it Works:
[![Roboflow: Streamlit Computer Vision App for Web Browser Deployment](https://img.youtube.com/vi/NXQ2Ktrh2BY/0.jpg)](https://www.youtube.com/watch?v=NXQ2Ktrh2BY)

## 💻 run locally
Complete these steps with your console/terminal
```
# clone the repository
git clone https://github.com/roboflow/streamlit-web-app.git
```
```
# navigate to the root directory
cd streamlit-web-app
cd streamlit
```
```
# set up your python environment and activate it
python3 -m venv venv
source venv/bin/activate
```
```
# install the requirements
pip install -r requirements.txt
```
```
# run the app
streamlit run Home.py
```

### Troubleshooting:
* For Mac users: be sure that you have [Homebrew](https://brew.sh/) installed.
* Unable to resolve wheel for `av` or `aiortc` packages: Install `pkg-config` by executing `brew install pkg-config` in your Terminal
* If you wish to process video streams with Streamlit apps, be sure to also have `ffmpeg` installed: after installing Homebrew, execute `brew install ffmpeg` in your Terminal

-- Check here for more on `ffmpeg` installation: https://github.com/roboflow/video-inference#requirements
* Ensure that you have `opencv-python-headless` installed in your environment, instead of `opencv-python`

* Unable to install the `av` package with `pip`? Try executing `conda install av -c conda-forge` in your Terminal

-- Note: [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is required for this method. Be sure that `conda-forge` has been made available in your channels. https://conda-forge.org/docs/user/introduction.html#how-can-i-install-packages-from-conda-forge

## Next Steps:
1. Customize the app further for your use case: https://docs.streamlit.io
2. Create another custom app or solution with your model
3. Create new models. Need inspiration? [Try searching Roboflow Universe](https://universe.roboflow.com)
4. Getting false detections? Model not performing to your expectations? Try Active Learning (Smart Sampling) and improving your Dataset Health:
* Rapid Testing and Deployment Iteration with [Roboflow Train](https://docs.roboflow.com/train) and the [Deploy Tab](https://blog.roboflow.com/deploy-tab)
* [How to Use Dataset Health Check to Improve Model Quality](https://www.youtube.com/watch?v=aUFz6P4dtk4)
* [What is Active Learning?](https://blog.roboflow.com/what-is-active-learning/) | [Why You Should Implement Active Learning](https://blog.roboflow.com/pip-install-roboflow)
  * Active Learning with Roboflow's Python Package: https://docs.roboflow.com/python/active-learning

### Increase Your Computer Vision Knowledge:
* https://roboflow.com/learn
#### Object Detection
[![Roboflow ML1M: What is Object Detection](https://img.youtube.com/vi/FM4-Pvrvo14/0.jpg)](https://www.youtube.com/watch?v=FM4-Pvrvo14)

#### Image Classification
[![Roboflow ML1M: What is Image Classification](https://img.youtube.com/vi/0swerezO3EQ/0.jpg)](https://www.youtube.com/watch?v=0swerezO3EQ)

#### [Instance] Segmentation
[![Roboflow ML1M: What is Instance Segmentation](https://img.youtube.com/vi/jLwWaN_2Omo/0.jpg)](https://www.youtube.com/watch?v=jLwWaN_2Omo)

#### [Object Detection vs. Image Classification vs. Keypoint Detection](https://blog.roboflow.com/object-detection-vs-image-classification-vs-keypoint-detection/)

#### [Semantic Segmentation vs. Instance Segmentation: Explained](https://blog.roboflow.com/difference-semantic-segmentation-instance-segmentation/)

## 🐞 bugs & 🦸 contribution

If you notice that any of the code is not working properly, create a [bug report](https://github.com/roboflow/streamlit-web-app/issues) 
and let us know.

If you have an idea for new functionality we should do, create a [feature request](https://github.com/roboflow/streamlit-web-app/issues).
If you feel up to the task and want to create a Pull Request yourself, please feel free to do so.

We are here for you, so don't hesitate to reach out here, or on our [Community Forum](https://github.com/roboflow/streamlit-web-app/issues).
