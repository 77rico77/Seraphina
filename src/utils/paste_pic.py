import cv2
import os
import numpy as np
from tqdm import tqdm
import uuid
from src.utils.videoio import save_video_with_watermark

def paste_pic(video_path, pic_path, crop_info, new_audio_path, full_video_path, extended_crop=False):
    if not os.path.isfile(pic_path):
        raise ValueError('pic_path must be a valid path to video/image file')

    # Load the full image
    if pic_path.split('.')[-1] in ['jpg', 'png', 'jpeg']:
        full_img = cv2.imread(pic_path)
    else:
        video_stream = cv2.VideoCapture(pic_path)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        still_reading, frame = video_stream.read()
        video_stream.release()
        if not still_reading:
            raise ValueError('Failed to read the video file')
        full_img = frame

    frame_h, frame_w = full_img.shape[:2]

    # Load the video frames
    video_stream = cv2.VideoCapture(video_path)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    crop_frames = []
    while True:
        still_reading, frame = video_stream.read()
        if not still_reading:
            break
        crop_frames.append(frame)
    video_stream.release()

    if len(crop_info) != 3:
        raise ValueError("Invalid crop_info format")

    r_w, r_h = crop_info[0]
    clx, cly, crx, cry = crop_info[1]
    lx, ly, rx, ry = map(int, crop_info[2])

    if extended_crop:
        oy1, oy2, ox1, ox2 = cly, cry, clx, crx
    else:
        oy1, oy2, ox1, ox2 = cly + ly, cly + ry, clx + lx, clx + rx

    # Debug prints
    print(f"full_img dimensions: {frame_w}x{frame_h}")
    print(f"Init ROI: ox1={ox1}, ox2={ox2}, oy1={oy1}, oy2={oy2}")

    # Clamp ROI values to be within the bounds of the image
    ox1 = max(0, min(ox1, frame_w - 1))
    ox2 = max(0, min(ox2, frame_w))
    oy1 = max(0, min(oy1, frame_h - 1))
    oy2 = max(0, min(oy2, frame_h))
    print(f"Clamped ROI: ox1={ox1}, ox2={ox2}, oy1={oy1}, oy2={oy2}")

    tmp_path = str(uuid.uuid4()) + '.mp4'
    out_tmp = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_w, frame_h))
    for crop_frame in tqdm(crop_frames, 'seamlessClone:'):
        p = cv2.resize(crop_frame.astype(np.uint8), (ox2 - ox1, oy2 - oy1))

        mask = 255 * np.ones(p.shape, p.dtype)
        location = ((ox1 + ox2) // 2, (oy1 + oy2) // 2)
        gen_img = cv2.seamlessClone(p, full_img, mask, location, cv2.NORMAL_CLONE)
        out_tmp.write(gen_img)

    out_tmp.release()

    save_video_with_watermark(tmp_path, new_audio_path, full_video_path, watermark=False)
    os.remove(tmp_path)