#-------------------------------------------------------------------------------
# Name:        Object bounding box label tool
# Purpose:     Label object bboxes for ImageNet Detection data
# Author:      Michael Laskey
# Created:     07/05/2017

#
#-------------------------------------------------------------------------------
from __future__ import division
from Tkinter import *
import tkMessageBox
from PIL import Image, ImageTk
import ttk
import os
import glob
import cPickle as pickle
import configs.config_bed as cfg
import random
import IPython
import cv2

# colors for the bboxes
COLORS = ['red', 'blue', 'cyan', 'green', 'black']
# image sizes for the examples
SIZE = 256, 256

class QueryLabeler():
    def __init__(self):
        # set up the main frame
        self.parent = Tk()
        self.parent.title("LabelTool")
        self.frame = Frame(self.parent)
        self.frame.pack(fill=BOTH, expand=1)
        self.parent.resizable(width = FALSE, height = FALSE)

        # initialize global state
        self.imageDir = ''
        self.imageList= []
        self.egDir = ''
        self.egList = []
        self.outDir = ''
        self.cur = 0
        self.total = 0
        self.category = 0
        self.imagename = ''
        self.labelfilename = ''
        self.tkimg = None
        self.currentLabelclass = ''
        self.cla_can_temp = []
        self.classcandidate_filename = 'class.txt'

        # initialize mouse state
        self.STATE = {}
        self.STATE['click'] = 0
        self.STATE['x'], self.STATE['y'] = 0, 0

        # reference to bbox
        self.bboxIdList = []
        self.bboxId = None
        self.bboxList = []
        self.hl = None
        self.vl = None

        # ----------------- GUI stuff ---------------------
        # dir entry & load
        self.label = Label(self.frame, text = "Image Dir:")
        self.label.grid(row = 0, column = 0, sticky = E)
        self.entry = Entry(self.frame)
        self.entry.grid(row = 0, column = 1, sticky = W+E)
        self.ldBtn = Button(self.frame, text = "Load", command = self.get_label)
        self.ldBtn.grid(row = 0, column = 2,sticky = W+E)

        # main panel for labeling
        self.mainPanel = Canvas(self.frame, cursor='tcross')
        self.mainPanel.bind("<Button-1>", self.mouseClick)
        self.mainPanel.bind("<Motion>", self.mouseMove)
        self.parent.bind("<Escape>", self.cancelBBox)  # press <Espace> to cancel current bbox
  

        # for ease of use
        self.parent.bind("f", lambda event: self.delBBox())
        self.parent.bind("q", lambda event: self.class_key_update("q"))
        self.parent.bind("w", lambda event: self.class_key_update("w"))
        self.parent.bind("e", lambda event: self.class_key_update("e"))
        self.parent.bind("r", lambda event: self.class_key_update("r"))
        self.parent.bind("t", lambda event: self.class_key_update("t"))

        self.mainPanel.grid(row = 1, column = 1, rowspan = 4, sticky = W+N)

        # choose class
        self.classname = StringVar()
        self.classcandidate = ttk.Combobox(self.frame,state='readonly',textvariable=self.classname)
        self.classcandidate.grid(row=1,column=2)

        self.cla_can_temp = cfg.CLASSES
        print self.cla_can_temp

        self.classcandidate['values'] = self.cla_can_temp
        self.classcandidate.current(0)
        self.currentLabelclass = self.classcandidate.get() #init
        self.btnclass = Button(self.frame, text = 'ComfirmClass', command = self.setClass)
        self.btnclass.grid(row=2,column=2,sticky = W+E)

        # showing bbox info & delete bbox
        self.lb1 = Label(self.frame, text = 'Bounding boxes:')
        self.lb1.grid(row = 3, column = 2,  sticky = W+N)
        self.listbox = Listbox(self.frame, width = 22, height = 12)
        self.listbox.grid(row = 4, column = 2, sticky = N+S)
        self.btnDel = Button(self.frame, text = 'Delete', command = self.delBBox)
        self.btnDel.grid(row = 5, column = 2, sticky = W+E+N)
        self.btnClear = Button(self.frame, text = 'ClearAll', command = self.clearBBox)
        self.btnClear.grid(row = 6, column = 2, sticky = W+E+N)

        # control panel for image navigation
        self.ctrPanel = Frame(self.frame)
        self.ctrPanel.grid(row = 7, column = 1, columnspan = 2, sticky = W+E)
        self.prevBtn = Button(self.ctrPanel, text='<< Prev', width = 10, command = self.prevImage)
        self.prevBtn.pack(side = LEFT, padx = 5, pady = 3)
        self.nextBtn = Button(self.ctrPanel, text='Send Command', width = 10, command = self.sendCommand)
        self.nextBtn.pack(side = LEFT, padx = 5, pady = 3)
        self.progLabel = Label(self.ctrPanel, text = "Progress:     /    ")
        self.progLabel.pack(side = LEFT, padx = 5)
        self.tmpLabel = Label(self.ctrPanel, text = "Go to Image No.")
        self.tmpLabel.pack(side = LEFT, padx = 5)
        self.idxEntry = Entry(self.ctrPanel, width = 5)
        self.idxEntry.pack(side = LEFT)
        self.goBtn = Button(self.ctrPanel, text = 'Go', command = self.gotoImage)
        self.goBtn.pack(side = LEFT)


        # example pannel for illustration
        self.egPanel = Frame(self.frame, border = 10)
        self.egPanel.grid(row = 1, column = 0, rowspan = 5, sticky = N)
        self.tmpLabel2 = Label(self.egPanel, text = "Examples:")
        self.tmpLabel2.pack(side = TOP, pady = 5)
        self.egLabels = []
        for i in range(3):
            self.egLabels.append(Label(self.egPanel))
            self.egLabels[-1].pack(side = TOP)

        # display mouse position
        self.disp = Label(self.ctrPanel, text='')
        self.disp.pack(side = RIGHT)

        self.frame.columnconfigure(1, weight = 1)
        self.frame.rowconfigure(4, weight = 1)

        self.label_count = 0

        # for debugging
##        self.setImage()
##        self.loadDir()

    def loadDir(self, dbg = False):
        s = 000
##        if not os.path.isdir(s):
##            tkMessageBox.showerror("Error!", message = "The specified dir doesn't exist!")
##            return
        # get image list
        self.imageDir = cfg.IMAGE_PATH
        # print self.imageDir
        #print self.category

        self.imageList = glob.glob(os.path.join(self.imageDir, '*.png'))
        # print self.imageList
        if len(self.imageList) == 0:
            print 'No .JPG images found in the specified dir!'
            return

        #get label list
        self.labelDir = cfg.LABEL_PATH
        self.labelListOrig = glob.glob(os.path.join(self.labelDir, '*.p'))
        self.labelList = [os.path.split(label)[-1].split('.')[0] for label in self.labelListOrig]

        #remove already labeled images
        self.imageList = [img for img in self.imageList if os.path.split(img)[-1].split('.')[0] not in self.labelList]
        #to review already labelled ones
        # self.imageList = [img for img in self.imageList if os.path.split(img)[-1].split('.')[0] in self.labelList]

        # default to the 1st image in the collection
        self.cur = 1
        self.total = len(self.imageList)
        print("total is " + str(self.total))
         # set up output dir
        self.outDir = cfg.LABEL_PATH

        # load example bboxes
        #self.egDir = os.path.join(r'./Examples', '%03d' %(self.category))
        self.egDir = os.path.join(r'./Examples/demo')
        # print os.path.exists(self.egDir)
        if not os.path.exists(self.egDir):
            return
        filelist = glob.glob(os.path.join(self.egDir, '*.png'))
        self.tmp = []
        self.egList = []
        random.shuffle(filelist)
        for (i, f) in enumerate(filelist):
            if i == 3:
                break
            im = Image.open(f)
            r = min(SIZE[0] / im.size[0], SIZE[1] / im.size[1])
            new_size = int(r * im.size[0]), int(r * im.size[1])
            self.tmp.append(im.resize(new_size, Image.ANTIALIAS))
            self.egList.append(ImageTk.PhotoImage(self.tmp[-1]))
            self.egLabels[i].config(image = self.egList[-1], width = SIZE[0], height = SIZE[1])

        self.loadImage()
        print '%d images loaded from %s' %(self.total, s)


    def get_label(self):

        
        self.current_image = self.cam.read_color_data()

        self.tkimg = ImageTk.PhotoImage(Image.fromarray(self.current_image))
        
        self.mainPanel.config(width = max(self.tkimg.width(), 400), height = max(self.tkimg.height(), 400))
        self.mainPanel.create_image(0, 0, image = self.tkimg, anchor=NW)
        self.progLabel.config(text = "%04d/%04d" %(self.cur, self.total))

        # load labels
        self.clearBBox()
        self.imagename = 'frame_'+str(self.label_count)
        labelname = self.imagename + '.p'
        self.labelfilename = os.path.join(self.outDir, labelname)
        bbox_cnt = 0
        self.lock = True
        
        



    def loadImage(self):
        # load image
        imagepath = self.imageList[self.cur - 1]

        self.img = Image.open(imagepath)
        self.tkimg = ImageTk.PhotoImage(self.img)
        self.mainPanel.config(width = max(self.tkimg.width(), 400), height = max(self.tkimg.height(), 400))
        self.mainPanel.create_image(0, 0, image = self.tkimg, anchor=NW)
        self.progLabel.config(text = "%04d/%04d" %(self.cur, self.total))

        # load labels
        self.clearBBox()
        self.imagename = os.path.split(imagepath)[-1].split('.')[0]
        labelname = self.imagename + '.p'
        self.labelfilename = os.path.join(self.outDir, labelname)
        bbox_cnt = 0
        # print('testing loading of label')
        # print(self.labelfilename)
        # print(os.path.exists(self.labelfilename))
        if os.path.exists(self.labelfilename):
            label = pickle.load( open(self.labelfilename, "rb") )
            for obj in label['objects']:
                curr_bbox = list(obj['box_index'])
                curr_bbox.append(cfg.CLASSES[obj['num_class_label']])
                tmp = curr_bbox
                self.bboxList.append(tuple(curr_bbox))

                tmpId = self.mainPanel.create_rectangle(int(tmp[0]), int(tmp[1]), \
                                                        int(tmp[2]), int(tmp[3]), \
                                                        width = 2, \
                                                        outline = COLORS[(len(self.bboxList)-1) % len(COLORS)])
                # print tmpId
                self.bboxIdList.append(tmpId)
                self.listbox.insert(END, '%s : (%d, %d) -> (%d, %d)' %(tmp[4],int(tmp[0]), int(tmp[1]), \
                												  int(tmp[2]), int(tmp[3])))
                self.listbox.itemconfig(len(self.bboxIdList) - 1, fg = COLORS[(len(self.bboxIdList) - 1) % len(COLORS)])
          

    def saveImage(self):

        label_data = {}
        label_data['num_labels'] = len(self.bboxList)
        objects = []
        for bbox in self.bboxList:
            obj = {}

            obj['box'] = bbox[0:4]
            obj['class'] = cfg.CLASSES.index(bbox[4])
            objects.append(obj)

        label_data['objects'] = objects

        self.label_data = label_data
        pickle.dump(label_data, open(cfg.LABEL_PATH+self.imagename+'.p','wb'))
        cv2.imwrite(cfg.IMAGE_PATH+self.imagename+'.png',self.current_image)
        print 'Image No. %d saved' %(self.cur)

    def mouseClick(self, event):
        if self.STATE['click'] == 0:
            self.STATE['x'], self.STATE['y'] = event.x, event.y
        else:
            x1, x2 = min(self.STATE['x'], event.x), max(self.STATE['x'], event.x)
            y1, y2 = min(self.STATE['y'], event.y), max(self.STATE['y'], event.y)
            self.bboxList.append((x1, y1, x2, y2, self.currentLabelclass))
            self.bboxIdList.append(self.bboxId)
            self.bboxId = None
            self.listbox.insert(END, '%s : (%d, %d) -> (%d, %d)' %(self.currentLabelclass,x1, y1, x2, y2))
            self.listbox.itemconfig(len(self.bboxIdList) - 1, fg = COLORS[(len(self.bboxIdList) - 1) % len(COLORS)])
        self.STATE['click'] = 1 - self.STATE['click']

    def mouseMove(self, event):
        self.disp.config(text = 'x: %d, y: %d' %(event.x, event.y))
        if self.tkimg:
            if self.hl:
                self.mainPanel.delete(self.hl)
            self.hl = self.mainPanel.create_line(0, event.y, self.tkimg.width(), event.y, width = 2)
            if self.vl:
                self.mainPanel.delete(self.vl)
            self.vl = self.mainPanel.create_line(event.x, 0, event.x, self.tkimg.height(), width = 2)
        if 1 == self.STATE['click']:
            if self.bboxId:
                self.mainPanel.delete(self.bboxId)
            self.bboxId = self.mainPanel.create_rectangle(self.STATE['x'], self.STATE['y'], \
                                                            event.x, event.y, \
                                                            width = 2, \
                                                            outline = COLORS[len(self.bboxList) % len(COLORS)])

    def cancelBBox(self, event):
        if 1 == self.STATE['click']:
            if self.bboxId:
                self.mainPanel.delete(self.bboxId)
                self.bboxId = None
                self.STATE['click'] = 0

    def delBBox(self):
        sel = self.listbox.curselection()
        if len(sel) != 1 :
            return
        idx = int(sel[0])
        self.mainPanel.delete(self.bboxIdList[idx])
        self.bboxIdList.pop(idx)
        self.bboxList.pop(idx)
        self.listbox.delete(idx)

    def clearBBox(self):
        for idx in range(len(self.bboxIdList)):
            self.mainPanel.delete(self.bboxIdList[idx])
        self.listbox.delete(0, len(self.bboxList))
        self.bboxIdList = []
        self.bboxList = []

    def prevImage(self, event = None):
        self.saveImage()
        if self.cur > 1:
            self.cur -= 1
            self.loadImage()

    def sendCommand(self, event = None):
        self.saveImage()
        self.lock = False
        if self.cur < self.total:
            self.cur += 1
            self.loadImage()

    def gotoImage(self):
        idx = int(self.idxEntry.get())
        if 1 <= idx and idx <= self.total:
            self.saveImage()
            self.cur = idx
            self.loadImage()

    def setClass(self):
    	self.currentLabelclass = self.classcandidate.get()
    	print 'set label class to :',self.currentLabelclass

    def class_key_update(self, class_label):
        mapping = {"q": ("mustard", 0), "w": ("syrup", 1), "e": ("salad_dressing", 2),
            "r": ("oatmeal", 3), "t": ("mayoniase", 4)}
        self.currentLabelclass = mapping[class_label][0]
        self.classcandidate.current(mapping[class_label][1])
        print 'set label class to :',self.currentLabelclass

    def run(self,cam):
        #self.parent.resizable(width =  True, height = True)
        self.cam = cam
        #self.current_image = img
        print "running"
        self.parent.mainloop()



if __name__ == '__main__':
    root = Tk()
    tool = LabelTool(root)
    root.resizable(width =  True, height = True)
    root.mainloop()
