import numpy as np
import cv2

def get_bbx(heatmap: np.array, threshold: int, min_detected_area: int):
    _, binary = cv2.threshold(heatmap, threshold, 255, cv2.THRESH_BINARY)
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    bbxes = []
    for i, stat in enumerate(stats):
        if i != 0:
            area = stat[cv2.CC_STAT_WIDTH]*stat[cv2.CC_STAT_HEIGHT]
            if min_detected_area < area:
                bbxes.append(
                    {
                        'left': stat[cv2.CC_STAT_LEFT],
                        'top': stat[cv2.CC_STAT_TOP],
                        'width': stat[cv2.CC_STAT_WIDTH],
                        'height': stat[cv2.CC_STAT_HEIGHT]
                    }
                )
    judge = None
    if len(bbxes) == 0:
        judge = 'OK'
    else:
        judge = 'NG'
    return binary, bbxes, judge

def draw_bbx(img: np.array, bbxes):
    c_img = img.copy()
    for bbx in bbxes:
        cv2.rectangle(c_img, (bbx['left'], bbx['top']), (bbx['left']+bbx['width'], bbx['top'] + bbx['height']), (255, 0, 255), 1)   
    return c_img

def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x