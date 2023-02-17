from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pickle
import keras
from keras.models import load_model
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import imutils


def get_captcha_image(driver):
    driver.find_element_by_id('vLoginPic').click()
    time.sleep(5)
    im = driver.get_screenshot_as_png()
    im = Image.open(BytesIO(im))
    image = driver.find_element_by_id('vLoginPic')
    print(image.location)
    im_loc = image.location
    left = im_loc['x']
    top = im_loc['y']
    right = im_loc['x'] + 102
    bottom = im_loc['y'] + 38

    im = im.crop((left, top, right, bottom))
    im.save('captcha.png')
    return im


def resize_to_fit(image, width, height):
    """
    A helper function to resize an image to fit within a given size
    :param image: image to resize
    :param width: desired width in pixels
    :param height: desired height in pixels
    :return: the resized image
    """

    (h, w) = image.shape[:2]

    if w > h:
        image = imutils.resize(image, width=width)
    else:
        image = imutils.resize(image, height=height)

    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)

    image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
        cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    return image


def binarize_image(img_path, target_path, threshold):
    """Binarize an image."""
    image_file = Image.open(img_path)
    image = image_file.convert('L')  # convert image to monochrome
    image = np.array(image)
    image = binarize_array(image, threshold)
    cv2.imwrite(target_path, image)

    return image


def binarize_array(numpy_array, threshold=150):
    """Binarize a numpy array."""
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[0])):
            if numpy_array[i][j] > threshold:
                numpy_array[i][j] = 255
            else:
                numpy_array[i][j] = 0
    return numpy_array


def switch_letter_background_color(image_array):
    unique, counts = np.unique(image_array, return_counts=True)
    if len(unique) == 2 and counts[0] < counts[1]:
        for i in range(len(image_array)):
            for j in range(len(image_array[0])):
                if image_array[i][j] == 255:
                    image_array[i][j] = 0
                else:
                    image_array[i][j] = 255
        return image_array


def get_prediction(model):
    image_file = 'captcha.png'
    binarized = binarize_image(image_file, 'captcha_b.png', 150)
    binarized = switch_letter_background_color(binarized)
    contours, _ = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gray = cv2.cvtColor(binarized, cv2.COLOR_GRAY2RGB)
    letter_image_regions = []
    for contour in contours:
        print(cv2.boundingRect(contour))
        (x, y, w, h) = cv2.boundingRect(contour)
        if w * h < 80:
            pass
        else:
            letter_image_regions.append((x, y, w, h))
    if len(letter_image_regions) != 4:
        return 0
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
    predictions = []
    # loop over the letters
    for letter_bounding_box in letter_image_regions:
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box

        # Extract the letter from the original image with a 2-pixel margin around the edge
        letter_image = gray[y:y + h, x:x + w]
        letter_image = cv2.cvtColor(letter_image, cv2.COLOR_BGR2GRAY)

        # Re-size the letter image to 20x20 pixels to match training data
        letter_image = resize_to_fit(letter_image, 20, 20)

        # Turn the single image into a 4d list of images to make Keras happy
        letter_image = np.expand_dims(letter_image, axis=2)
        letter_image = np.expand_dims(letter_image, axis=0)

        # Ask the neural network to make a prediction
        prediction = model.predict(letter_image)

        # Convert the one-hot-encoded prediction back to a normal letter
        letter = lb.inverse_transform(prediction)[0]
        predictions.append(letter)


    # Print the captcha's text
    captcha_text = "".join(predictions)
    print("predict CAPTCHA : {}".format(captcha_text))
    return captcha_text


load 模型
MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)
    model = load_model(MODEL_FILENAME)

options = Options()
options.add_argument("--disable-notifications")

web_address = "https://xxx"
chrome = webdriver.Chrome('./chromedriver', chrome_options=options)
chrome.get(web_address)

# switch to iframe
iframe = chrome.find_element_by_tag_name('iframe')
chrome.switch_to.default_content()
chrome.switch_to.frame(iframe)
iframe_source = chrome.page_source

# 登入輸入帳密並submit
close_js = chrome.find_element_by_id('js-noticle-close').click()
username = chrome.find_element_by_id('username')
password = chrome.find_element_by_id('passwd')
username.send_keys('johhnytest3')
password.send_keys('Abc789')
time.sleep(5)
submit = chrome.find_element_by_id('submit-log').submit()
time.sleep(5)


captcha_im = get_captcha_image(chrome)
captcha_text = get_prediction(model)
rmNum = chrome.find_element_by_id('rmNum')
rmNum.send_keys(captcha_text)

chrome.find_element_by_id('js-pic-confirm-btn').click()

# 若驗證碼錯誤，關掉alert，再重新點選login，取得screenshot
WebDriverWait(chrome, 10).until(EC.alert_is_present())
chrome.switch_to.alert.accept()
submit = chrome.find_element_by_id('submit-log').submit()


captcha_text = 0
while captcha_text == 0:
    print('Im sleeping')
    time.sleep(3)
    chrome.find_element_by_id('vLoginPic').click()
    captcha_im = get_captcha_image(chrome)
    captcha_text = get_prediction(model)
    rmNum = chrome.find_element_by_id('rmNum')
    rmNum.send_keys(captcha_text)

    chrome.find_element_by_id('js-pic-confirm-btn').click()
    try:
        WebDriverWait(chrome, 10).until(EC.alert_is_present())
        chrome.switch_to.alert.accept()
        submit = chrome.find_element_by_id('submit-log').submit()
    except:
        print('no alert')


