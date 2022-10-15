from tkinter.filedialog import *
ImageLoc = askopenfilename(initialdir="/", title="이미지 선택", filetypes=(("png 파일", "*.png"), ("모든 파일", "*.*")))
print(ImageLoc)