import cv2,time,os,tensorflow as tf
import numpy as np

from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(123)

class detector:
    def __init__(self):
        pass

    def readClass(self,clas_path):
        with open(clas_path,'r') as f:
            self.classlist = f.read().splitlines()

        self.colorlist = np.random.uniform(low =0,high=255,size = (len(self.classlist), 3))

   

    def model_dowload(self,url):

        file_name = os.path.basename(url)
        self.model_name = file_name[:file_name.index('.')]


        self.dir = "./models"

        os.makedirs(self.dir,exist_ok=True)

        get_file(fname=file_name,origin=url,cache_dir=self.dir,cache_subdir="check",extract=True)



    def loadmodel(self):
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(self.dir,"check",self.model_name,"saved_model"))




    def create_box(self,image):
        input = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        input = tf.convert_to_tensor(input, dtype=tf.uint8)
        input = input[tf.newaxis,...]

        detect = self.model(input)

        box = detect['detection_boxes'][0].numpy()
        class_index = detect['detection_classes'][0].numpy().astype(np.int32)
        class_scores = detect['detection_scores'][0].numpy()

        hei , wei , chan = image.shape
        
        box = tf.image.non_max_suppression(box,confidence,max_output_size=50,iou_threshold=0.5,score_threshold=0.5)

        if len(box)!=0:
            for i in box:
                box = tuple(box[i].tolist())
                confidence = 100*class_scores[i]
                index = class_index[i]

                name = self.classlist[index]
                colo = self.colorlist[index]

                display_text = '{}: {}%'.format(name,confidence)

                ymin, xmin, ymax, xmax = box

                xmin, xmax, ymin, ymax = (int(xmin*wei),int(xmax*wei),int(ymin*hei),int(ymax*hei))

                cv2.rectangle(image, (xmin,xmax), (ymin,ymax), color=colo,thickness=3)
                cv2.putText(image,display_text,(xmin,ymin-10),cv2.FONT_HERSHEY_COMPLEX,1,colo,3)
        return image



    def video_detection(self,video):

        video_char = cv2.VideoCapture(video)

        if (video_char.isOpened()== False): 
            print("Error opening video stream or file")
            return

        (success, image) = video_char.read()


        start_time = 0
        while success:
            time_now = time.time()

            fps  = 1/(time_now - start_time)
            start_time = time_now

            around_image = self.create_box(image)

            cv2.putText(around_image,"FPS -" + str(int(fps)), (20,70), cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),2)
            cv2.imshow("result",around_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            (success,image) = video_char.read()
        cv2.destroyAllWindows()



    




        
