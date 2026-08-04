pip install opencv-python

# Install opencv (required for running the demo)
pip3 install -U opencv-python

# Install detectron2
python tools/setup_detectron2.py

# Install other dependencies
pip3 install git+https://github.com/cocodataset/panopticapi.git
pip3 install git+https://github.com/mcordts/cityscapesScripts.git
pip3 install -r requirements.txt
pip3 install imutils colormap