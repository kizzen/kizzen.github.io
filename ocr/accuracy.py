from ocr import detectNum
import os

directory = 'pics/'
accuracy_lst = []
count = 0
for fname in os.listdir(directory):
	if fname.lower().endswith('.jpg'):
		count+=1
		true_class = fname.split('_')[1][-2:]
		try:
			pred = detectNum(directory + fname)
		except:
			pred = fname
		if true_class==pred:
			accuracy_lst.append(1)
		else:
			accuracy_lst.append(0)
print('OCR accuracy: ',round(sum(accuracy_lst)/count*100,2),'%')