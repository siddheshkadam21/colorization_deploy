import numpy as np
import cv2
import streamlit as st
from PIL import Image

st.title("Image Colorization")


def Image_colorization(image2):
    net = cv2.dnn.readNetFromCaffe("colorization_deploy_v2.prototxt", "colorization_release_v2.caffemodel")
    pts = np.load('pts_in_hull.npy')
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)

    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype='float32')]

    scaled = image2.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    ab = cv2.resize(ab, (image2.shape[1], image2.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)

    colorized = (255 * colorized).astype("uint8")
    return colorized


uploaded_file = st.file_uploader("Choose a Black and White Image", type=["jpg", "jpeg", "png"])
# net = load_model()
if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file))
    image = cv2.resize(image, (224, 224))

    placeholder = st.beta_columns(2)
    placeholder[0].image(image)
    out_image = Image_colorization(image)
    placeholder[1].image(out_image)
