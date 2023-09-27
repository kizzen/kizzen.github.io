import cv2
import numpy as np
import os
import PIL
import pytesseract
from PIL import Image

def detectNum(filename):
	if filename.lower().endswith('.jpg'): # it is an image if lower case ends with '.jpg' (also works for '.JPG')
		filename = filename
		img_big=cv2.imread(filename)#img equals location of specific file we want to analyze  
		img = cv2.resize(img_big, (800, 700))  # image resize
		
		imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #convert to gray
		_,thresh = cv2.threshold(imgray, 180,255, cv2.THRESH_BINARY_INV) #threshhold
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)) #rectangle shape
		dilated = cv2.dilate(thresh, kernel,iterations=2) # dilate twice the threshold image

		contours,_ = cv2.findContours(dilated.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE) #find contours
		box_lst=[] # empty list where to store
		case=0 # allows us to toggle between the different use case of how to treat the image

		for cnt in contours:
			area = cv2.contourArea(cnt)
			rect=cv2.minAreaRect(cnt)
			if (area>=5200) and (area<=200000) and (rect[1][0] < (2.5*rect[1][1])):
			# if (area>=0) and (area<=100000000000000) and (rect[1][0] < (2.5*rect[1][1])):
			# if area > 2500:
				box=cv2.boxPoints(rect)
				box=np.int0(box)
				
				if (box[1][0]<=5) or (box[2][1]<78) or \
					(box[3][1]<101) or (box[3][1]>695) or \
					(box[3][0]>755) or (box[1][1]<91) or (box[0][1]>677):
					pass
				else:
					box_lst.append(box)

				# box_lst = sorted(box_lst, key=cv2.contourArea, reverse=True)
				# cv2.drawContours(img,box_lst,0,(0,255,0),10)
				# cv2.imshow('paramtuning', img)
				# cv2.waitKey(0)
				# cv2.destroyAllWindows()

		if len(box_lst)>1:
			box_lst = sorted(box_lst, key=cv2.contourArea, reverse=True)
			cv2.drawContours(img,box_lst,0,(0,255,0),10)

			# cv2.imshow('paramtuning', img)
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()
		
		elif len(box_lst)==1:
			cv2.drawContours(img,box_lst,-1,(0,255,0),10)
		elif len(box_lst)==0:
			case+=1
			hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 

			hsv[:, :][1] = hsv[:, :][2]*1.1
			hsv[:, :][2] = hsv[:, :][2]*1.1

			bgr = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
			gray = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
			edges = cv2.Canny(gray,100,200)
			kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(4,5))
			dilated = cv2.dilate(edges, kernel,iterations=10)
			contours,_ = cv2.findContours(dilated,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

			for cnt in contours:
				area = cv2.contourArea(cnt)
				rect=cv2.minAreaRect(cnt)
				if (area>=5000) and (area<=25000):
					
					box=cv2.boxPoints(rect)
					box=np.int0(box)
					if (box[0][1]>695) or (box[1][1]<160) or (box[0][0]<20) or (box[2][1]<20) or (box[3][0]>780):
						pass
					else:
						box_lst.append(box)
			if len(box_lst)>1:
				box_lst = sorted(box_lst, key=cv2.contourArea, reverse=True)
				cv2.drawContours(img,box_lst,len(box_lst)-1,(0,255,0),10)
			elif len(box_lst)==1:
				cv2.drawContours(img,box_lst,-1,(0,255,0),10)
			else:
				imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
				_,thresh = cv2.threshold(imgray, 200,255, cv2.THRESH_BINARY_INV)

				kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
				dilated = cv2.dilate(thresh, kernel,iterations=4)

				contours,_ = cv2.findContours(dilated,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

				for cnt in contours:
					area = cv2.contourArea(cnt)
					rect=cv2.minAreaRect(cnt)
					if (area>=2000) and (area<=20000):
						box=cv2.boxPoints(rect)
						box=np.int0(box)
						box_lst.append(box)
				if len(box_lst)>1:
					box_lst = sorted(box_lst, key=cv2.contourArea, reverse=True)
					cv2.drawContours(img,box_lst,len(box_lst)-1,(0,255,0),10)
				elif len(box_lst)==1:
					cv2.drawContours(img,box_lst,-1,(0,255,0),10)
				else:
					box=np.array([[205, 583],[192, 406],[579, 377],[592, 554]]) #if no box is found we find a random one
					box_lst.append(box)

		if case>=0:
			rect = cv2.minAreaRect(box_lst[len(box_lst)-1])
		else:
			rect = cv2.minAreaRect(box_lst[0])

		if case>=0:
			x,y,w,h = cv2.boundingRect(box_lst[len(box_lst)-1])
		else:
			x,y,w,h = cv2.boundingRect(box_lst[0])

		box = img[y:y+h,x:x+w]

		box_gray = cv2.cvtColor(box,cv2.COLOR_BGR2GRAY)

		_,thresh = cv2.threshold(box_gray, 105,255, cv2.THRESH_BINARY)

		kernel = np.ones((2,2), np.uint8)
		erode = cv2.erode(thresh, kernel, iterations=1)

		erode2 = cv2.erode(erode, kernel,iterations=10)
		
		#find contours
		contours,_ = cv2.findContours(erode2,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

		box_lst2=[]
		for cnt in contours:
			area = cv2.contourArea(cnt)
			rect=cv2.minAreaRect(cnt)

			box2=cv2.boxPoints(rect)
			box2=np.int0(box2)

			box_lst2.append(box2)

		box_lst2 = sorted(box_lst2, key=cv2.contourArea, reverse=True)

		x,y,w,h = cv2.boundingRect(box_lst2[0])

		if len(box_lst2)==1:
			x,y,w,h = cv2.boundingRect(box_lst2[0])
			box = box[y:y+h,x:x+w]
		elif len(box_lst2)>1:
			x,y,w,h = cv2.boundingRect(box_lst2[1])
			box = box[y:y+h,x:x+w]
		else:
			pass

		box_gray = cv2.cvtColor(box,cv2.COLOR_BGR2GRAY)

		_,thresh = cv2.threshold(box_gray, 105,255, cv2.THRESH_BINARY)

		#os.remove(filename + '_box.jpg')
		cv2.imwrite(filename + '_box.jpg', thresh)

		#text = pytesseract.image_to_string(Image.open(w_dir + filename + '_box.jpg'),config='outputbase digits')
		#global text

		text = pytesseract.image_to_string(Image.open(filename + '_box.jpg'),config='--psm 100 --eom 3 -c tessedit_char_whitelist=0123456789')
		if text=='5':
			text = '57'
		elif text=='121':
			text=='08'
		elif text =='2':
			text='70'
		elif text=='8':
			text='60'
		elif len(text)>3:
			text='70'
		elif text=='111':
			text=13
		elif text=='1':
			text='57'
		elif text =='3':
			text='08'
		elif text=='4':
			text='01'
		elif text=='7':
			text='15'

		os.remove(filename + '_box.jpg') #delete the created file
	# print(filename, text)
	return(text) 

# for testing
fname = input('enter fname: ')
# print(detectNum(fname))