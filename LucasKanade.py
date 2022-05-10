import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image


LucasKanade_parameters = dict(winSize = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
feature_parameters = dict(maxCorners = 1000, qualityLevel = 0.25, minDistance = 7, blockSize = 7)

source = str(input('Input source: '))

class Algorithm:
    def __init__(self):
        self.capture = cv2.VideoCapture(source)
        self.points = []
        self.circle_radius = 2
        self.track_len = 8
        self.color = (0,255,0)
        self.interval = 5
        self.frame_index = 0

        if (self.capture.isOpened() == False):
            print('Error loading source')
            exit()
        else:
            print('Streaming video')

    def main(self):
        while True:
            ret, frame = self.capture.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            visual = frame.copy()

            if len(self.points) > 0:
                frame0, frame1 = self.previous_gray, frame_gray
                point0 = np.float32([track[-1] for track in self.points]).reshape(-1, 1, 2)
                point1, status, error = cv2.calcOpticalFlowPyrLK(frame0, frame1, point0, None, **LucasKanade_parameters)
                point0_reverse, status, error = cv2.calcOpticalFlowPyrLK(frame1, frame0, point1, None, **LucasKanade_parameters)
                match = abs(point0-point0_reverse).reshape(-1, 2).max(-1)
                control = match < 1
                cpoints = []
                for track, (x, y), red_flag in zip(self.points, point1.reshape(-1, 2), control):
                    if not red_flag:
                        continue
                    track.append((x, y))
                    if len(track) > self.track_len:
                        del track[0]
                    cpoints.append(track)
                    cv2.circle(visual, (int(x), int(y)), self.circle_radius, self.color, -1)
                self.points = cpoints
                cv2.polylines(visual, [np.int32(track) for track in self.points], False, self.color)

            if self.frame_index % self.interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(track[-1]) for track in self.points]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                point_pos = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_parameters)
                if point_pos is not None:
                    for x, y in np.float32(point_pos).reshape(-1, 2):
                        self.points.append([(x, y)])

            pil_im = Image.fromarray(visual)
            pil_draw = ImageDraw.Draw(pil_im)
            pil_font = ImageFont.truetype('assets/Roboto-Regular.ttf', 24)

            pil_draw.text((10, 80), 'FPS: ' + str(int(self.capture.get(cv2.CAP_PROP_FPS))), font=pil_font)
            pil_draw.text((10, 120), 'Motion points: ' + str(len(self.points)), font=pil_font)

            frame_display = np.array(pil_im)

            self.frame_index += 1
            self.previous_gray = frame_gray
            cv2.imshow('Lucas Kanade Algorithm', frame_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('Stream interrupted')
                break


Algorithm().main()
cv2.destroyAllWindows()
