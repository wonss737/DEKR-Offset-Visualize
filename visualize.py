import cv2
from PIL import Image
import numpy as np

def draw_flow(img, flow, color=(255, 255, 255), step=8):
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
    heatmap_keypoint = np.array(heatmap_keypoint, dtype=np.uint8)
    heatmap_keypoint = cv2.applyColorMap(heatmap_keypoint, cv2.COLORMAP_JET)
    return heatmap_keypoint * 0.7 + img * 0.3

def save_image(image, heatmaps, offset, image_name, final_output_dir):

    image = image.clone().cpu().numpy()
    heatmaps = heatmaps.clone().mul(255).clamp(0, 255).cpu().numpy()
    offset = offset.clone().cpu().numpy()

    height = heatmaps.shape[2]
    width = heatmaps.shape[3]

    image = np.squeeze(image.transpose(0, 2, 3, 1), axis=0)
    image = ((image - np.min(image)) / np.ptp(image)) * 255
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    grid = np.zeros((height * 7, width * 5, 3))
    a=c=0
    for i in range(35):
        height_begin = height * (i // 5)
        width_begin = width * (i % 5)
        image_ = image.copy()
        showImage = np.zeros((height, width, 3))
        if i < 18:
            heatmap = np.squeeze(heatmaps[:, a, :, :], axis=0)
            showImage = draw_heatmap(image_, heatmap)
            a+=1
        else :
            center_th = np.mean(heatmaps[0, 17, :, :] )
            offset[:, c:c+2, :, :][:, :, heatmaps[0, 17, :, :] < center_th] = 0
            offset_keypoint = np.squeeze(offset[:, c:c+2, :, :], axis=0)
            showImage = draw_flow(image_, np.transpose(offset_keypoint, axes=(1,2,0)), color=(0, 0, 255))
            c+=2
        grid[height_begin:height_begin+height, width_begin:width_begin+width] = showImage

    cv2.imwrite(final_output_dir + '/results/' + image_name, grid)
