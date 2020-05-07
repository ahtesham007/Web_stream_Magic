from flask import Response,render_template,Flask
import threading
import argparse
import time
import warnings
import cv2
import time
import numpy as np
warnings.filterwarnings("ignore")


outputFrame = None
lock = threading.Lock()
ds_factor=0.6
# initialize a flask object
app = Flask(__name__)
vs = cv2.VideoCapture(0)
time.sleep(2.0)

@app.route("/")
def index():
    return render_template('index.html')

def detect():
    global vs, outputFrame, lock
    time.sleep(3)
    count = 0
    background = 0

    ## Capture the background in range of 60
    for i in range(60):
        ret, background = vs.read()
    background = np.flip(background, axis=1)

    while True:
        ret, img = vs.read()
        if not ret:
            break
        count += 1
        img = np.flip(img, axis=1)

        ## Convert the color space from BGR to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        ## Generat masks to detect red color
        lower_red = np.array([0, 120, 50])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([170, 120, 70])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)

        mask1 = mask1 + mask2

        ## Open and Dilate the mask image
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

        ## Create an inverted mask to segment out the red color from the frame
        mask2 = cv2.bitwise_not(mask1)

        ## Segment the red color part out of the frame using bitwise and with the inverted mask
        res1 = cv2.bitwise_and(img, img, mask=mask2)

        ## Create image showing static background frame pixels only for the masked region
        res2 = cv2.bitwise_and(background, background, mask=mask1)

        ## Generating the final output and writing
        finalOutput = cv2.addWeighted(res1, 1, res2, 1, 0)
        frame = cv2.resize(finalOutput, None, fx=ds_factor, fy=ds_factor)
        with lock:
            outputFrame = frame.copy()


def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock

	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue

			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# ensure the frame was successfully encoded
			if not flag:
				continue

		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed", )
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    #ap.add_argument("-i", "--ip", type=str, required=True,help="ip address of the device")
    #ap.add_argument("-o", "--port", type=int, required=True,help="ephemeral port number of the server (1024 to 65535)")
    #ap.add_argument("-f", "--frame-count", type=int, default=32,
                   # help="# of frames used to construct the background model")
    args = vars(ap.parse_args())

    # start a thread that will perform motion detection
    t = threading.Thread(target=detect)
    t.daemon = True
    t.start()

    # start the flask app
    app.run(debug=True,threaded=True, use_reloader=False)

# release the video stream pointer
vs.release()
