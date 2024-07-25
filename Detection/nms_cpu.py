import numpy as np

def iou(boxA, boxB):
    # 计算两个边界框的交集坐标
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # 计算交集面积
    interArea = max(0, xB - xA) * max(0, yB - yA)
    
    # 计算每个边界框的面积
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    # 计算并集面积
    unionArea = boxAArea + boxBArea - interArea
    
    # 计算IoU
    iou = interArea / unionArea if unionArea != 0 else 0
    
    return iou

def nms(boxes, scores, iou_threshold):
    picked = []  # 存储被选择的边界框索引
    indexes = np.argsort(scores)[::-1]  # 按分数降序排列索引
    
    while len(indexes) > 0:
        current = indexes[0]
        picked.append(current)  # 选择当前最高分的边界框
        indexes = indexes[1:]  # 移除当前最高分的索引
        
        # 检查剩余边界框与当前选择框的IoU，如果大于阈值则抑制
        indexes = [i for i in indexes if iou(boxes[current], boxes[i]) <= iou_threshold]
    
    return picked


# 假设boxes和scores是模型预测的边界框和分数
boxes = np.array([[50, 50, 100, 100], [60, 60, 110, 110], [200, 200, 300, 300]])
scores = np.array([0.9, 0.75, 0.8])

# 设置IoU阈值
iou_threshold = 0.5

# 执行NMS
picked_boxes = nms(boxes, scores, iou_threshold)

print("Selected box indices:", picked_boxes)
