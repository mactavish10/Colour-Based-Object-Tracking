import cv2
import numpy as np
from collections import deque

def val(x) :
	# print('Current value of selected trackbar = ',x)
	pass


def createtrackbar() : #function to create trackbar
	cv2.namedWindow('Track Bar')

	for i in [' Low',' High'] :
		for j in ['H' , 'S' , 'V'] :
			cv2.createTrackbar(str(j)+str(i) , 'Track Bar' , 0,255,val)


def trackbarposition() : #function to get trackbar position

	intensities = []

	for i in [' Low',' High'] :
		for j in ['H' , 'S' , 'V'] :
			val = cv2.getTrackbarPos(str(j)+str(i),'Track Bar')
			intensities.append(val)
	return intensities





def main() :

	cap = cv2.VideoCapture(0) 

	createtrackbar()
	points_list = deque(maxlen = 64)

	while True :
		ret , frame = cap.read()

		if not ret :
			print('Error opening camera/video!')

		frame = cv2.resize(frame , (1366,768))

		frame = cv2.flip(frame,1) #flip the frame, not necessary

		

		blur = cv2.GaussianBlur(frame , (11,11) , 0) #smoothen the image before contour hunt
		
		hsv = cv2.cvtColor(blur , cv2.COLOR_BGR2HSV) #convert to hsv colorspace, which is more effective for segmenting out colors 

		# h,s,v = cv2.split(hsv);
                
  #       # if equalizeH:
		# h = cv2.equalizeHist(h);
  #               # if equalizeS:
		# s = cv2.equalizeHist(s);
  #               # if equalizeV:
		# v = cv2.equalizeHist(v);
		# hsv = cv2.merge([h,s,v]);

		min1 , max1 = [] , []

		values = trackbarposition() #get current trackbar position, i.e., h,s,v channel minimum and maximum values respectively
		min1 = values[0:3] #h_low , s_low , v_low
		max1 = values[3:6] #h_high , s_high , v_high

		# print(min1 , max1)

		img_thresh = cv2.inRange(hsv , tuple(min1) , tuple(max1)) #retain objects according to min1 and max1 values 

		# cv2.imshow('Thresholded Image' , img_thresh)

		# kernel = cv2.getStructuringElement(cv2.MORPH_RECT , (5,5))

		# mask = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
		# mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

		mask = cv2.erode(img_thresh , None , iterations=2)
		mask = cv2.dilate(mask , None , iterations=2)

		contours , _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #find contours on the processed image
		center = None #initialize center as none for now
		if len(contours) > 0 :
			cnts = sorted(contours , key = lambda x : cv2.contourArea(x) , reverse = True) #sort contours in descending order, on basis of contour area
 			# cnts = contours

			for c in cnts[0:1] : #iterate through the contours(for now, just one, you can try all of them too!)
				M = cv2.moments(c)
				center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) #get center of the contour
				cv2.circle(frame, center, 8, (0, 0, 255), -1) #plot a circle at the center
				# cv2.drawContours(frame , c , -1 , (0,0,0) , 4)

		points_list.appendleft(center) #append the center positions so as to show the path taken by the object
		# print(points_list)

		for i in range(1 , len(points_list)) :
			if points_list[i-1] is None or points_list[i] is None :
				print('continue')
				continue
			thickness = int(np.sqrt(64 / float(i + 1)) * 1.5)
			cv2.line(frame, points_list[i - 1], points_list[i], (0, 0, 255), thickness) #draw path followed by the object

		cv2.imshow('Frame' , frame)
		cv2.imshow('Mask' , mask)

		if cv2.waitKey(1) & 0xFF == ord('q') :
			break

	cv2.destroyAllWindows()
	cap.release()

if __name__ == "__main__" :
	main()