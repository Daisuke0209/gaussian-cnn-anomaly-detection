from typing import List, Dict
import numpy as np
import cv2

def get_bbx(heatmap : np.array, threshold : int = 0, min_detected_area : int = 0):
    """get_bbx
    get bounding boxes from heatmap 

    Parameters
    -------
    heatmap : np.array
        Heatmap which is made by GaussianCnnPredictor
    threshold : int
        Areas below threshold will not be bouneding boxes
    min_detected_area : 
        Bouneding boxes below min_detected_area will not be returned

    Returns
    -------
    binary : np.array
        Binarization of heatmap
    bbxes : List[Dict]
        Bounding boxes in the abnormal area
    judge : str
        OK or NG
    """
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

def draw_bbx(img : np.array, bbxes : List[Dict]):
    """draw_bbx
    draw bounding boxes on input image  

    Parameters
    -------
    img : np.array
        Input image
    threshold : int
        Areas below threshold will not be bouneding boxes
    min_detected_area : 
        Bouneding boxes below min_detected_area will not be returned

    Returns
    -------
    c_img : np.array
        Image with bounding boxes
    """
    c_img = img.copy()
    for bbx in bbxes:
        cv2.rectangle(c_img, (bbx['left'], bbx['top']), (bbx['left']+bbx['width'], bbx['top'] + bbx['height']), (255, 0, 255), 1)   
    return c_img

def denormalization(image):
    """denormalization
    denormalize image

    Parameters
    -------
    img : np.array
        Input image
        
    Returns
    -------
    image : np.array
        denormalized image
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (((image.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return image