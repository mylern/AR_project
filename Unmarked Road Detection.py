import numpy as np
import cv2
from PIL import Image
import math

last_right 		= []
last_left  		= []
left_angles 	= []
right_angles	= []
l = []
r = [] 


# dir = r"ENTER FILE'S DIRECTORY HERE AND UNCOMMENT" # <<<<<<<<<<<<-------------------------------------------------------

videoName = [r'\Tests\NoMarkings_.mp4']	

#Controls
video_number = 0


Gaussian_x   = 25
Gaussian_y   = 25
Num_colours  = 2
size_y		 = 1.5
size_x		 = 7
threshold    = 20
maxgap		 = 5
linelen		 = 20




#Return the filename of video
def vname(i):
    return ''.join([dir,videoName[i]])


#Take an image in OpenCV format, convert to PIL format, 
# Quantize to number_of_colours and convert back to openCV format


def quantize(image, number_of_colours):

	array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	pil_im = Image.fromarray(array)
	pil_im = pil_im.quantize(number_of_colours).convert('L')#Convert to grayscale
	opencvImage = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

	return opencvImage




#Take an image and return binary edge detected image
def canny_edge_detector(image):

	#Perform Guassian Blur of specified size
	blur = cv2.GaussianBlur(image, (Gaussian_x, Gaussian_y), 0)  

	cv2.imshow("Original with blur", blur)

	quantized = quantize(blur, Num_colours)
	# quantized = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY)
	cv2.imshow("Quantized", quantized)

	#Perform canny edge detection on greyscale image
	canny = cv2.Canny(quantized, 50, 150) 

	return canny


#Create a specified shape in image
#Return only this part of the image

def region_of_interest(image): 

	height, width = image.shape 

	global size_y
	global size_x
	
	h1 = int(height/size_y)
	h2 = int(height- height/4)
	w1 = int(width/size_x)
	w2 = int(width-w1)

	#Create a polygon of specified shape
	shape = [(w1,h1), (w2, h1), (w2, h2), (w1, h2)] 
	
	polygons = np.array([ 
		shape
		]) 

	#Create an image same dimensions as before all zeros
	mask = np.zeros_like(image) 
	
	#Fill image will 1s where specified
	cv2.fillPoly(mask, polygons, 255) 
	
	#AND the given image with the ANDed image to create a mask image
	masked_image = cv2.bitwise_and(image, mask) 

	#Return the mask image
	return masked_image
	


#Take the slope and y intercept of a line and return the x,y values for the lines
def create_coordinates(image, line_parameters): 
	#If there are lines
	try:
		slope, intercept = line_parameters

	#Allow no values
	except TypeError:
	    slope, intercept = 0,0

	
	if slope == 0:
		x1 = y1 = x2 = y2 = 0
	else:
		#set y1 value to bottom of image
		y1 = image.shape[0] 
		#set y2 value to 2/5 up the image
		y2 = int(y1 * (3 / 5)) 
		#x1 is where line passes through the bottom of image
		x1 = int((y1 - intercept) / slope) 
		#x2 is where line passes through 2/5 of image
		x2 = int((y2 - intercept) / slope)
	
	#Return in order x1, y1, x2, y2
	return np.array([x1, y1, x2, y2])  


#Take two co-ordinates of line and return the absolute value of the angle made 
def angle(array):
	x1 = array[0]
	y1 = array[1]
	x2 = array[2]
	y2 = array[3]

	if x1 != 0:
		m = (y2-y1)/(x2-x1)
		return abs(int(math.degrees(math.atan(m))))
	else:
		return 0


def average(list):
	return sum(list)/len(list)	


#Take an array and return the average of the last four elements
def average_previous_angles(slope, array):
	if len(array) == 0:
		return slope
	elif len(array) < 2:
		return average(array[-len(array):])
	else:
		return average(array[-2:])


#Take an image and all straight edges that appear in this image,
#return the xy values for the left and right lines to be drawn

def average_slope_intercept(image, lines, i): 
	left_fit = [] 
	right_fit = [] 

	global left_angles, right_angles

	if lines is not None:
		for line in lines: 
			x1, y1, x2, y2 = line.reshape(4) 
			
			#Use numpy Polyfit to extract slope and intercept from co-ordinates
			parameters = np.polyfit((x1, x2), (y1, y2), 1) 
			slope = parameters[0] 
			intercept = parameters[1] 				

			#If the slope is left
			if slope < 0: 
				left_fit.append((slope, intercept)) 
			#If the slope is right
			if slope > 0: 
				right_fit.append((slope, intercept)) 


	global last_left
	global last_right#


	#If there are lines, use them, if not, use the previous frame's lines
	if left_fit:
		last_left  	= left_fit
	else:
		left_fit 	= last_left
	if right_fit:	
		last_right 	= right_fit
	else:
		right_fit 	= last_right

	global l,r
	
	#Get the average of the left and right lines
	left_fit_average = np.average(left_fit, axis = 0)
	right_fit_average = np.average(right_fit, axis = 0)

	#Create points from average lines
	left_line = create_coordinates(image, left_fit_average)
	right_line = create_coordinates(image, right_fit_average)

	#Find angles produced
	left = angle(left_line)	
	right = angle(right_line)

	if i < 5:
		left_angles.append(left)
		l = left_line
		
		right_angles.append(right)
		r = right_line
	
	#If the difference in the angles is less than 7 degrees different
	#Than the previous 4 frames, don't display it and use the previous
	#Frames lines instead
	#This avoids big changes in line angle
	else:
		diff = average_previous_angles(left, left_angles)
		if abs(diff-left) < 7:
			left_angles.append(left)
			l = left_line

		diff = average_previous_angles(right, right_angles)
		if abs(diff-right) < 7:
			right_angles.append(right)
			r = right_line

	#return the xy values for the left and right lines to be drawn
	return np.array([l, r])


#Take an image and return an image of same dimensions containing the given lines
def display_lines(image, lines):
	line_image = np.zeros_like(image)

	if lines is not None:
		for x1, y1, x2, y2 in lines:
			cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)

	return line_image


i = 0

#Declare name of video to process
name = vname(video_number)

cap = cv2.VideoCapture(name)
_, frame = cap.read()

#While frames in video
while(cap):

	#Create a canny image on the frame
	canny_image = canny_edge_detector(frame)
	cv2.imshow("Original", frame)
	cv2.imshow("Canny", canny_image)
	#Crop the canny image of the frame to the region of interest & display
	cropped_image = region_of_interest(canny_image)
	cv2.imshow("Cropped", cropped_image)

	#Find all the lines in the cropped cranny frame
	lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, threshold, np.array([]), minLineLength = linelen, maxLineGap = maxgap)
	
	#Calculate the average left and right line of the found lines
	averaged_lines = average_slope_intercept(frame, lines, i)

	#create an image with the average left and right lines on it
	line_image = display_lines(frame, averaged_lines) 

	#Add this image to the original & display the result
	combo_image = cv2.addWeighted(frame, 1, line_image, 0.7, 1)
	cv2.imshow("results", combo_image)

	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	i+=1
	_, frame = cap.read()


print(i)
cap.release() 
cv2.destroyAllWindows() 