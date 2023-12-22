import os
import streamlit as st
import numpy as np
import cv2
from streamlit_option_menu import option_menu
import tensorflow as tf
from PIL import Image

SIZE = 224

def load_image():
    
    uploaded_file = st.file_uploader(label='Pick an image to test',type=["png","jpg","jpeg"])
    if uploaded_file is not None:
        #image_data = uploaded_file.getvalue()
        
        #Saving upload
        with open(os.path.join("uploaded_img","1.jpg"),"wb") as f:
            f.write((uploaded_file).getbuffer())
            st.success("File Saved")
        return preprocess_img()
    else:
        return None

def preprocess_img():
    image_dat = cv2.imread("uploaded_img/1.jpg", cv2.IMREAD_COLOR)
    w, h = image_dat.shape[1], image_dat.shape[0]
    crop_width = 1000 if 1000<image_dat.shape[1] else image_dat.shape[0]
    crop_height = 1000 if 1000<image_dat.shape[0] else image_dat.shape[0] 
    mid_x, mid_y = int(w/2), int(h/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2) 
    crop_img = image_dat[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    image_dat = cv2.resize(crop_img, (SIZE, SIZE))
    image_dat = cv2.cvtColor(image_dat, cv2.COLOR_RGB2BGR)
    st.image(image_dat)
    #image_dat = image_dat.reshape(1,224,225,3)
    image_dat = np.reshape(image_dat,[1,224,224,3])
    image_dat = image_dat/255.0
    print(image_dat)
    return image_dat

@st.cache_resource    
def load_saved_model():
    #model = load_model('best_model_1.h5')
    model = tf.keras.models.load_model('best_model_1.h5')
    return model

def load_labels():
    labels_path = 'class.txt'
    labels_file = os.path.basename(labels_path)
    with open(labels_file, "r") as f:
        categories = [s.strip() for s in f.readlines()]
        return categories


def predict(model, categories, image):
    CAT = categories
    fc_prediction = model.predict(image)
    predicted_class = np.argmax(fc_prediction)
    print(predicted_class)
    st.write(CAT[predicted_class])
    st.write(fc_prediction)
    return predicted_class


def main():
    #hide_st_style = """
    #        <style>
            #MainMenu {visibility: hidden;}
    #        footer {visibility: hidden;}
    #        header {visibility: hidden;}
    #        </style>
    #        """
    #st.markdown(hide_st_style, unsafe_allow_html=True)
    logo_utm = Image.open('img/logo-utm-128.png')
    st.image(logo_utm, caption='')
    st.title('Model Demo Deteksi Kualitas Garam')
    
    selected2 = option_menu(None, ["Home", "Upload", "Kamera"], 
    icons=['house', 'cloud-upload', "camera"], 
    menu_icon="cast", default_index=0, orientation="horizontal")
    #selected2
    if selected2 == "Home":
        st.write("Halaman Utama")
        st.write("Pilih Upload untuk melakukan klasifikasi garam menggunakan data gambar")
        st.write("Pilih Kamera untuk melakukan klasifikasi garam menggunakan data gambar dari kamera")
        st.write("_Tim Kedaireka Garam UTM 2023_")

    elif selected2 == "Upload":
       image = load_image()
       if image is not None:
            result = st.button('Run on image')
            if result:
                model = load_saved_model()
                categories = load_labels()
                st.write('Calculating results...')
                predict(model, categories, image)
                #print(categories)

    elif selected2 == "Kamera":
       picture = st.camera_input("Take a picture")
       if picture:
          #st.image(picture)
          with open(os.path.join("uploaded_img","1.jpg"),"wb") as f:
            f.write((picture).getbuffer())
            st.success("File Saved")
            image = preprocess_img()
            result = st.button('Run on image')
            if result:
                model = load_saved_model()
                categories = load_labels()
                st.write('Calculating results...')
                predict(model, categories, image)


if __name__ == '__main__':
    main()