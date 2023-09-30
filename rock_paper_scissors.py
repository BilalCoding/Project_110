# To Capture Frame
import cv2
# To process image array
import numpy as np
# Import the tensorflow modules and load the model
import tensorflow as tf
model = tf.keras.models.load_model("keras_model.h5")
# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)

# Infinite loop
while True:
	# Reading / Requesting a Frame from the Camera 
	status , frame = camera.read()
	# If we were sucessfully able to read the frame
	if status:
		# Flip the frame
		frame = cv2.flip(frame, 1)
		# Resize the frame
		img = cv2.resize(frame, (224, 224))
		# Expand the dimensions
		test_img = np.array(img, dtype=np.float32)
		test_img = np.expand_dims(test_img, axis=0)
		# Normalize it before feeding to the model
		normalised_img = test_img/255.0
		# Get predictions from the model
		prediction = model.predict(normalised_img)
		print("Prediction:", prediction)
		# Displaying the frames captured
		cv2.imshow('feed', frame)

		# Waiting for 1ms
		code = cv2.waitKey(1)
		
		# If space key is pressed, break the loop
		if code == 32:
			break
# Release the camera from the application software
camera.release()
# Close the open window
cv2.destroyAllWindows()
