import cv2

#to take image input (RGB format)
img=cv2.imread("input2.png")
cv2.imwrite("1_original.png",img)

#to convert it to Grayscale Image
g_f= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imwrite("2_grayscale.png",g_f)

#to invert the colors of grayscaled imange
i=cv2.bitwise_not(g_f)
cv2.imwrite("3_invert.png",i)

#to make blur image from grayscaled and inverted image
b=cv2.GaussianBlur(i,(21,21),0)
cv2.imwrite("4_blur.png",b)

#to combine inverterd and blurred image
ib=cv2.bitwise_not(b)
cv2.imwrite("5_invertedblurry.png",ib)

#to make Final Sketch
sketch= cv2.divide(g_f,ib,scale=256.0)
cv2.imwrite("6_sketch.png",sketch)#output




