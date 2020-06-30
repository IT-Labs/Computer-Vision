from selenium import webdriver
from time import sleep
import argparse

pathToChromeDriver = r"path to you chrome driver exe"

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-images", type=int, default=500, help="# of images to download")
args = vars(ap.parse_args())

options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(pathToChromeDriver, chrome_options=options)

for i in range(0, args["num_images"]):

    driver.get('path to site')
    sleep(1)
    pathToSaveThePicture = r"path where you want to save the picture" + str(i) + ".png"
    image = driver.find_element_by_css_selector('img[id=CaptchaImage]').screenshot(pathToSaveThePicture)
    sleep(0.2)

driver.quit()