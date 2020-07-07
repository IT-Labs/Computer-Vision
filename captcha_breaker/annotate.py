from imutils import paths
import argparse
import imutils
import cv2
import os
import uuid

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input directory of images")
ap.add_argument("-a", "--annot", required=True, help="path to output directory of annotations")
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(os.path.dirname(__file__) + '\\' + args["input"]))
counts = {}

for (i, imagePath) in enumerate(imagePaths):
    print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))

    try:
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #cnts = cnts[0]
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:3]

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
                cv2.imshow("ROI", imutils.resize(roi1, width=28))
                key = cv2.waitKey(0)
                if key == ord("`"):
                    print("[INFO] ignoring character")
                    continue

                key = chr(key).upper()
                dirPath = os.path.sep.join(os.path.dirname(__file__) + '\\' + args["input", key])
                if not os.path.exists(dirPath):
                    os.makedirs(dirPath)

                count = counts.get(key, 1)
                p = os.path.sep.join([dirPath, "{}.png".format(uuid.uuid1())])
                cv2.imwrite(p, roi1)

                counts[key] = count + 1

            cv2.imshow("ROI", imutils.resize(roi, width=28))
            key = cv2.waitKey(0)
            if key == ord("`"):
                print("[INFO] ignoring character")
                continue

            key = chr(key).upper()
            dirPath = os.path.sep.join([os.path.dirname(__file__), args["annot"], key])
            if not os.path.exists(dirPath):
                os.makedirs(dirPath)

            count = counts.get(key, 1)
            p = os.path.sep.join([dirPath, "{}.png".format(str(count).zfill(6))])
            cv2.imwrite(p, roi)

            counts[key] = count + 1

    except KeyboardInterrupt:
        print("[INFO] manually leaving script")
        break

    except:
        print("[INFO] skipping image...")