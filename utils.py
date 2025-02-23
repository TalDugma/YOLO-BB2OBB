def xyxy2xywh(x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1
    x = x1 + w/2
    y = y1 + h/2
    return x, y, w, h

def xywh2xyxy(x,y,w,h):
    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2
    return x1, y1, x2, y2

def normallize_coords(x, y, w, h, img_w, img_h):
    x /= img_w
    y /= img_h
    w /= img_w
    h /= img_h
    return x, y, w, h

def denormallize_coords(x, y, w, h, img_w, img_h):
    x *= img_w
    y *= img_h
    w *= img_w
    h *= img_h
    return x, y, w, h

def normalize_obb_coords(x1, y1, x2, y2, x3, y3, x4, y4, img_w, img_h):
    x1 /= img_w
    y1 /= img_h
    x2 /= img_w
    y2 /= img_h
    x3 /= img_w
    y3 /= img_h
    x4 /= img_w
    y4 /= img_h
    return x1, y1, x2, y2, x3, y3, x4, y4

# def convert_to_frame_coords(x, y, w, h, window_x, window_y, frame_w, frame_h):
#     """
#     takes as input the denormalized coordinates of a bounding box inside a cropped window and returns the coordinates of the bounding box in the original frame
#     """
#     x += window_x
#     y += window_y
    
    