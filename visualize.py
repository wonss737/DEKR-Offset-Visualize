import cv2
from PIL import Image
import numpy as np

def draw_flow(img, flow, color=(255, 255, 255), step=32):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x - fx, y - fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    for line in lines:
        if np.sum(line[0] - line[1]) != 0:
            cv2.arrowedLine(img, tuple(line[0]), tuple(line[1]), color=color)
    return img

def draw_heatmap(img, heatmap_keypoint):
    heatmap_keypoint = cv2.cvtColor(heatmap_keypoint, cv2.COLOR_GRAY2BGR) * 255
    return heatmap_keypoint * 0.7 + img * 0.3

def save_image(image, heatmaps, masks, offset, offset_w, num, final_output_dir):

    image = image.clone().cpu().numpy()
    heatmaps = heatmaps.clone().cpu().numpy()
    masks = masks.clone().cpu().numpy()
    offset = offset.clone().cpu().numpy()
    offset_w = offset_w.clone().cpu().numpy()

    height = heatmaps.shape[2]
    width = heatmaps.shape[3]

    image = np.squeeze(image.transpose(0, 2, 3, 1), axis=0)
    image = ((image - np.min(image)) / np.ptp(image)) * 255
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    grid = np.zeros((height * 10, width * 7, 3))
    a=b=c=d=0
    for i in range(70):
        height_begin = height * (i // 7)
        width_begin = width * (i % 7)
        image_ = image.copy()
        showImage = np.zeros((height, width, 3))
        if i < 18:
            heatmap = np.squeeze(heatmaps[:, a, :, :], axis=0)
            showImage = draw_heatmap(image_, heatmap)
            a+=1
        elif i >= 18 and i < 36:
            mask = np.squeeze(masks[:, b, :, :], axis=0)
            showImage = draw_heatmap(image_, mask)
            b+=1
        elif i >= 36 and i < 53:
            offset_keypoint = np.squeeze(offset[:, c:c+2, :, :], axis=0)
            showImage = draw_flow(image_, np.transpose(offset_keypoint, axes=(1,2,0)), color=(0, 0, 255))
            c+=2
        else:
            offset_w_keypoint = np.sum(np.squeeze(offset_w[:, d:d+2, :, :], axis=0), axis=0) * 200
            showImage = draw_heatmap(image_, offset_w_keypoint)
            d+=2
        grid[height_begin:height_begin+height, width_begin:width_begin+width] = showImage

    cv2.imwrite(final_output_dir + '/results/image_%07d.png' % num, grid)
