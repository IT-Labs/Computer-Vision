from selenium import webdriver
from time import sleep
import argparse
import cv2
import numpy as np
import imutils
from imutils import paths
from imutils import contours
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from helper.utils.captchahelper import preprocess
from sklearn.preprocessing import LabelBinarizer

#get arguments
ap = argparse.ArgumentParser()
ap.add_argument("-u", "--url", required=True, help="url of the site to break the captcha")
ap.add_argument("-m", "--model", required=True, help="path to directory of the trained model")
#ap.add_argument("-n", "--num-images", type=int, default=500, help="# of images to download")
args = vars(ap.parse_args())

#set arguments
url = args["url"]
model = load_model(os.path.dirname(__file__) + '\\' + args["model"], compile=False)

#chrome driver path
currDir = os.path.dirname(__file__)
pathToChromeDriver = os.path.join(currDir, 'chromedriver')
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(pathToChromeDriver, chrome_options=options)

#get image
driver.get(url)
sleep(1)
image = driver.find_element_by_css_selector('img[id=CaptchaImage]').screenshot('captcha_image.png')
sleep(0.2)

#split image
image = cv2.imread('captcha_image.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:3]
cnts = contours.sort_contours(cnts)[0]

#set labels
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S'
, 'T', 'U', 'V', 'W', 'Y', 'Z'];
labels = np.array(labels)
lb = LabelBinarizer().fit(labels)

predictions = []
for c in cnts:
	(x, y, w, h) = cv2.boundingRect(c)
	areTwoLetters = False
	if w / h > 2.2:
		half_width = int(w / 2)
		roi = gray[y - 5:y + h + 5, x - 5:x + half_width + 5]
		roi1 = gray[y - 5:y + h + 5, x + half_width - 5:x + w + 5]
		areTwoLetters = True
	else:
		roi = gray[y - 5:y + h + 5, x - 5:x + w + 5]
	if areTwoLetters:
		image = preprocess(roi1, 28, 28)
	
	image = preprocess(roi, 28, 28)
	image = img_to_array(image)
	
	image = np.expand_dims(image, axis=0) / 255.0
	pred = model.predict(image, batch_size=32).argmax(axis=1)[0]
	predictions.append(lb.classes_[pred])
	
captcha_text = "".join(predictions);
print("[INFO] captcha: {}".format(captcha_text))
driver.find_element_by_id("CaptchaInputText").send_keys(captcha_text)
	
key = cv2.waitKey(0)
#driver.quit()
