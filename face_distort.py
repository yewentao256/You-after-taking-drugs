import time
import cv2
import dlib
import math
import numpy as np
from scipy.spatial import Delaunay
import os

# dlib检测器进行初步人脸检测
face_detector = dlib.get_frontal_face_detector()
# dlib基于ERT算法实现的人脸定位
predictor_path = 'model/shape_predictor_68_face_landmarks.dat'
shape_predictor = dlib.shape_predictor(predictor_path)
customer_img_path = 'resources/customer2.png'
addict_img_path = 'resources/addictor.png'
output_path = 'out'
face_thin_alpha = 1 # 数值越大瘦脸程度越大
cut_image = True
exposure_alpha = 1  # 防曝光，如果出现异常蓝点调大此数值，大于1的值可能导致出现色差
# TODO：用一种更柔和的方式除曝光


def detect_face_and_cut(img: np.ndarray) -> np.ndarray:
    """识别人脸并裁剪合适的部分。

    核心逻辑为找到人脸矩形并裁剪其周边地区。效果为提取出照片中的人脸部分

    Args:
        img (np.ndarray): 待处理图片

    Raises:
        RuntimeError: 图中有多张脸部或未检测到脸部

    Returns:
        np.ndarray: 裁剪后图片
    """
    # 转化为灰度图
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 找到人脸矩形
    rects = face_detector(img_gray, 0)

    # 只需要有一张脸
    if len(rects) == 1:
        rect = rects[0]
        # 计算矩形框大小
        height = rect.bottom() - rect.top()
        width = rect.right() - rect.left()
        # 新矩形的点计算，要么可以扩展，要么直接为0
        new_top, new_left = int(max(rect.top() - height * 0.5,
                                    0)), int(max(rect.left() - width * 0.5, 0))
        new_bottom, new_right = int(
            min(rect.bottom() + height * 0.5, img.shape[0])), int(
                min(rect.right() + width * 0.5, img.shape[1]))

        # 截取图片,注意第一个参数是高，第二个参数是宽
        img_rect = img[new_top:new_bottom, new_left:new_right]
        return img_rect

    raise RuntimeError(
        f'There are {len(rects)} faces in your image, expecting only 1')


def get_face_68_landmarks(img: np.ndarray) -> np.ndarray:
    """获取人脸68个特征点

    Args:
        img (np.ndarray): 需获取特征点的图像

    Returns:
        np.ndarray: 68特征点列表
    """
    # 转化为灰度图加速检测运算
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = face_detector(img_gray, 1)
    return np.array([[p.x, p.y]
                     for p in shape_predictor(img, rects[0]).parts()])


def local_translation_warp(img: np.ndarray, center_point: np.ndarray,
                           mouse_point: np.ndarray,
                           radius: float) -> np.ndarray:
    """Interactive Image Warping 局部图像扭曲算法——平移扭曲

    论文： http://www.gson.org/thesis/warping-thesis.pdf

    Args:
        img (np.ndarray): 待处理图像
        center_point (np.ndarray): 中心点，包含x坐标和y坐标。
            在瘦脸算法中，该中心点为脸部最左/最右点。
        mouse_point (np.ndarray): 原文中为鼠标终止点，这里的含义为边界端点，
            包含x坐标和y坐标。在瘦脸算法中，这个端点为鼻尖上边的一点
        radius (float): 自定义影响最大半径，超过最大半径的扭曲会被忽略。
            影响最大半径小于中心点和终止点的距离才会起作用。
            也可以理解为某种意义上的strength，值越大扭曲效果越明显

    Returns:
        np.ndarray: 处理后的图像
    """
    img_copy = img.copy()

    height, width, _ = img.shape
    # 优化：计算需要处理的矩形范围，只在该范围内遍历。该优化提升比较大
    left, right = int(max(center_point[0] - radius, 0)), \
        int(min(center_point[0] + radius, width))
    bottom, high = int(max(center_point[1] - radius, 0)), \
        int(min(center_point[1] + radius, height))

    # 计算公式中的|m-c|^2， r^2
    mc_distance_pow = (mouse_point[0] - center_point[0])**2 + (
        mouse_point[1] - center_point[1])**2
    radius_pow = radius**2

    for i in range(left, right):
        for j in range(bottom, high):
            # 计算当前点与中心点的距离
            distance = (i - center_point[0])**2 + (j - center_point[1])**2

            # 判断该点是否在最大影响距离中，在才处理，否则不处理
            if distance < radius_pow:
                # 套用公式求出映射位置
                tmp = ((radius_pow - distance) /
                       (radius_pow - distance + mc_distance_pow))**2
                x = i - tmp * (mouse_point[0] - center_point[0])
                y = j - tmp * (mouse_point[1] - center_point[1])

                # 需要根据双线性插值法得到原图像对应的值
                value = BilinearInsert(img, x, y)
                # 改变当前点的值,注意索引是高, 宽
                img_copy[j, i] = value

    return img_copy


def BilinearInsert(image: np.ndarray, x: float, y: float) -> np.ndarray:
    """逆向映射双线性插值法，传入需要处理原图像的x、y坐标，求出对应邻近点加权后值

    Args:
        image (np.ndarray): 原图像
        x (float): 坐标x
        y (float): 坐标y

    Raises:
        RuntimeError: 输入非三通道图像报错

    Returns:
        np.ndarray: 加权后值value
    """
    _, _, c = image.shape
    if c == 3:
        x1, y1 = int(x), int(y)
        x2, y2 = x1 + 1, y1 + 1
        # 依次计算邻近各起作用的值，最后累加即所需值
        part1 = image[y1, x1] * (x2 - x) * (y2 - y)
        part2 = image[y1, x2] * (x - x1) * (y2 - y)
        part3 = image[y2, x1] * (x2 - x) * (y - y1)
        part4 = image[y2, x2] * (x - x1) * (y - y1)
        insert_value = part1 + part2 + part3 + part4
        return insert_value.astype(np.int8)

    raise RuntimeError('BilinearInsert中输入了非三通道图像')


def face_thin(img: np.ndarray, alpha: float = 1) -> np.ndarray:
    """ 对输入图像进行瘦脸

    Args:
        img (np.ndarray): 需要瘦脸的图像
        alpha (float, optional): 瘦脸参数，越大瘦脸效果越夸张

    Returns:
        np.ndarray: 瘦脸后的图像
    """
    # 获取关键点
    landmarks = get_face_68_landmarks(img)

    left_landmark = landmarks[3]  # 形如[195 119]，脸最左端特征点。
    left_landmark_down = landmarks[5]  # 脸部左下特征点
    right_landmark = landmarks[13]  # 脸部最右端特征点
    right_landmark_down = landmarks[11]  # 脸部右下特征点
    middle_landmark = landmarks[30]  # 鼻尖往上一点的特征点

    # 左边瘦脸最大影响半径为：
    # 脸最左端特征点到脸部左下特征点的距离（选其他的也ok，这里实测效果比较好）
    r_left = math.sqrt((left_landmark[0] - left_landmark_down[0])**2 +
                       (left_landmark[1] - left_landmark_down[1])**2)

    # 右边瘦脸最大影响半径为：脸最右端特征点到脸部右下特征点的距离
    r_right = math.sqrt((right_landmark_down[0] - right_landmark[0])**2 +
                        (right_landmark_down[1] - right_landmark[1])**2)

    # 瘦左边脸
    customer_thin_image = local_translation_warp(img, left_landmark,
                                                 middle_landmark,
                                                 r_left * alpha)
    # 瘦右边脸
    customer_thin_image = local_translation_warp(customer_thin_image,
                                                 right_landmark,
                                                 middle_landmark,
                                                 r_right * alpha)

    return customer_thin_image


def landmarks_add_8_points(image: np.ndarray,
                           points: np.ndarray) -> np.ndarray:
    """加入图片四个顶点和四条边的中点用于三角剖分

    Args:
        image (np.ndarray): 图片
        points (np.ndarray): 现有特征点

    Returns:
        np.ndarray: 加入八个特征点后的特征点矩阵
    """
    x, y = image.shape[1] - 1, image.shape[0] - 1
    points = points.tolist()
    points.append([0, 0])
    points.append([x // 2, 0])
    points.append([x, 0])
    points.append([x, y // 2])
    points.append([x, y])
    points.append([x // 2, y])
    points.append([0, y])
    points.append([0, y // 2])
    return np.array(points)


def affine_transform(input_image: np.ndarray, input_triangle: list,
                     output_triangle: list, size: tuple) -> np.ndarray:
    """传入转换前三角形和转换后三角形求出变换矩阵，再进行常规仿射变换

    Args:
        input_image (np.ndarray): 需处理的图像
        input_triangle (list): 转换前三角
        output_triangle (list): 转换后三角
        size (tuple): 图像大小

    Returns:
        np.ndarray: 变换后的图像
    """
    affine_matrix = cv2.getAffineTransform(np.float32(input_triangle),
                                           np.float32(output_triangle))

    result = cv2.warpAffine(input_image,
                            affine_matrix, (size[0], size[1]),
                            None,
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)
    return result


def morph_one_triangle(bottom_img: np.ndarray, mask_img: np.ndarray,
                       morph_img: np.ndarray, bottom_tri: list, mask_tri: list,
                       morph_tri: list, alpha: float) -> None:
    """传入底图、mask图和融合图的一组三角形坐标，进行三角变形与Alpha 混合

    Args:
        bottom_img (np.ndarray): 底图
        mask_img (np.ndarray): mask图
        morph_img (np.ndarray): 融合图（结果）
        bottom_tri (list): 底图的一个三角形
        mask_tri (list): mask图的一个三角形
        morph_tri (list): 融合图的一个三角形
        alpha (float): 融合参数
    """
    # 计算三角形的边界框，float32
    # 返回值形如[100,100,50,50]，表示左上角点为100,100，两条边长度分别为50
    bottom_tri_rect, mask_tri_rect = cv2.boundingRect(np.float32(
        [bottom_tri])), cv2.boundingRect(np.float32([mask_tri]))
    morph_tri_rect = cv2.boundingRect(np.float32([morph_tri]))

    # 存放处理好后的三角形信息的列表
    bottom_tri_deal, mask_tri_deal, morph_tri_deal = [], [], []

    # 对三角形中每个点处理，减去矩形左上角的点，得到归一至左上角的三角形
    for i in range(3):
        morph_tri_deal.append(((morph_tri[i][0] - morph_tri_rect[0]),
                               (morph_tri[i][1] - morph_tri_rect[1])))
        bottom_tri_deal.append(((bottom_tri[i][0] - bottom_tri_rect[0]),
                                (bottom_tri[i][1] - bottom_tri_rect[1])))
        mask_tri_deal.append(((mask_tri[i][0] - mask_tri_rect[0]),
                              (mask_tri[i][1] - mask_tri_rect[1])))

    # 在图形中裁剪出三角形所在矩形
    bottom_img_rect = bottom_img[bottom_tri_rect[1]:bottom_tri_rect[1] +
                                 bottom_tri_rect[3],
                                 bottom_tri_rect[0]:bottom_tri_rect[0] +
                                 bottom_tri_rect[2]]
    mask_img_rect = mask_img[mask_tri_rect[1]:mask_tri_rect[1] +
                             mask_tri_rect[3],
                             mask_tri_rect[0]:mask_tri_rect[0] +
                             mask_tri_rect[2]]

    # 对裁剪出的矩形图像，根据三角形应用仿射变换
    size = (morph_tri_rect[2], morph_tri_rect[3])
    bottom_affine_img = affine_transform(bottom_img_rect, bottom_tri_deal,
                                         morph_tri_deal, size)
    mask_affine_img = affine_transform(mask_img_rect, mask_tri_deal,
                                       morph_tri_deal, size)

    # 线性加权求和
    img_rect = (1.0 - alpha) * bottom_affine_img + alpha * mask_affine_img

    # 生成蒙版
    # cv2.fillConvexPoly填充凸多边形：参数:待处理图像，凸多边形顶点，颜色，line type
    mask = np.zeros((morph_tri_rect[3], morph_tri_rect[2], 3),
                    dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(morph_tri_deal), (1, 1, 1), 16)

    # 对于morph_img的局部地区应用蒙版
    morph_img[morph_tri_rect[1]:morph_tri_rect[1] + morph_tri_rect[3],
              morph_tri_rect[0]:morph_tri_rect[0] + morph_tri_rect[2]] = \
        morph_img[morph_tri_rect[1]:morph_tri_rect[1] + morph_tri_rect[3],
                  morph_tri_rect[0]:morph_tri_rect[0] + morph_tri_rect[2]] * \
        (1 - mask) + img_rect * mask


def triangle_face_morph(bottom_img: np.ndarray,
                        mask_img: np.ndarray,
                        landmarks_bottom: np.ndarray,
                        landmarks_mask: np.ndarray,
                        alpha: float = 0.5) -> np.ndarray:
    """三角融合，本质线性相加：M(x,y)=(1-α)I(x,y)+αJ(x,y)


    Args:
        bottom_img (np.ndarray): 底图
        mask_img (np.ndarray): 融合图
        landmarks_bottom (np.ndarray): 底图特征点
        landmarks_mask (np.ndarray): mask图特征点
        alpha (float, optional): 融合alpha参数，默认为0.5.

    Returns:
        np.ndarray: 融合图
    """
    # 底图和mask添加8个特征点
    landmarks_bottom = landmarks_add_8_points(bottom_img, landmarks_bottom)
    landmarks_mask = landmarks_add_8_points(mask_img, landmarks_mask)

    # alpha融合得到融合图特征点
    landmarks_morph = (1 - alpha) * landmarks_bottom + alpha * landmarks_mask

    # uint8转换为float32保留精度
    bottom_img, mask_img = np.float32(bottom_img), np.float32(mask_img)

    morph_img = np.zeros(bottom_img.shape, dtype=bottom_img.dtype)

    # Delaunay 三角剖分，返回可以组合成三角形的点索引
    triangles = Delaunay(landmarks_morph).simplices

    # 对每个三角形处理
    for triangle in triangles:
        point_1, point_2, point_3 = triangle[0], triangle[1], triangle[2]

        # 利用点索引取出坐标，生成形如[array([231, 308]), array([230, 308]),
        # array([215, 296])]等点列表，这些点可以构成三角形
        bottom_triangle = [
            landmarks_bottom[point_1], landmarks_bottom[point_2],
            landmarks_bottom[point_3]
        ]
        mask_triangle = [
            landmarks_mask[point_1], landmarks_mask[point_2],
            landmarks_mask[point_3]
        ]
        morph_triangle = [
            landmarks_morph[point_1], landmarks_morph[point_2],
            landmarks_morph[point_3]
        ]

        morph_one_triangle(bottom_img, mask_img, morph_img, bottom_triangle,
                           mask_triangle, morph_triangle, alpha)

    return np.uint8(morph_img)


def merge_img(bottom_img: np.ndarray,
              morph_img: np.ndarray,
              landmarks_bottom: np.ndarray,
              exposure_alpha: float = 1.1) -> np.ndarray:
    """高斯模糊底图和融合图，实现初步融合(颜色校正)后，泊松融合图像

    Args:
        bottom_img (np.ndarray): 底图
        morph_img (np.ndarray): 融合图
        landmarks_bottom (np.ndarray): 底图特征点
        exposure_alpha (float, optional): 防曝光修正系数,
            如果发现处理时有较多蓝色斑点则调大此数值. Defaults to 1.1.

    Returns:
        np.ndarray: 结果图
    """
    # 求出滤波核大小，np.linalg.norm计算范数
    kernel = int(0.4 * np.linalg.norm(
        np.mean(landmarks_bottom[36:42], axis=0) -
        np.mean(landmarks_bottom[42:48], axis=0)))
    if kernel % 2 == 0:
        kernel += 1

    # 应用高斯模糊
    bottom_img_blur = cv2.GaussianBlur(bottom_img, (kernel, kernel), 0)
    morph_img_blur = cv2.GaussianBlur(morph_img, (kernel, kernel), 0)

    # 防止除0错误
    morph_img_blur += (128 * (morph_img_blur <= 1.0)).astype(np.uint8)
    morph_img = np.uint8(
        morph_img.astype(np.float32) * bottom_img_blur.astype(np.float32) /
        morph_img_blur.astype(np.float32) / exposure_alpha)

    # 创建并处理面部蒙版
    face_mask = np.zeros(bottom_img.shape, dtype=bottom_img.dtype)
    cover = list(range(0, 27))  # 前27个特征点恰好包括整个面部轮廓
    cv2.fillConvexPoly(face_mask, cv2.convexHull(landmarks_bottom[cover]),
                       (255, 255, 255))  # 填充多边形，实现蒙版
    r = cv2.boundingRect(landmarks_bottom)  # 计算脸部边界框
    center = (r[0] + int(r[2] / 2), r[1] + int(r[3] / 2))  # 计算脸部中心

    # 对蒙版均值滤波平滑，使结果更加柔和
    face_mask = cv2.blur(face_mask, (15, 10), center)

    # 泊松融合
    result = cv2.seamlessClone(morph_img, bottom_img, face_mask, center,
                               cv2.NORMAL_CLONE)

    return result


def affine_transform_and_change_shape(img: np.ndarray,
                                      landmarks_target: np.ndarray,
                                      shape_target: tuple) -> np.ndarray:
    """仿射变换+调整形状

    运用普式分析法计算仿射变换矩阵，再应用仿射变换并调整图片大小，

    Args:
        img (np.ndarray): 需要调整的图片
        landmarks_target (np.ndarray): 底图特征点
        shape_target (tuple): 需调整的形状

    Returns:
        np.ndarray: 应用仿射变换后的新遮罩
    """
    # Procrustes analysis：通过旋转/平移等方式，使第一个向量点尽可能对齐
    # 第二个向量点，数据体现为最小二乘法距离和最小。
    landmarks_img = np.matrix(get_face_68_landmarks(img)).astype(np.float64)
    landmarks_target = np.matrix(landmarks_target).astype(np.float64)

    # 减均值
    c1, c2 = np.mean(landmarks_target, axis=0), np.mean(landmarks_img, axis=0)
    landmarks_target -= c1
    landmarks_img -= c2

    # 除标准差
    s1, s2 = np.std(landmarks_target), np.std(landmarks_img)
    landmarks_target /= s1
    landmarks_img /= s2

    # singular Value Decomposition奇异值分解
    U, S, Vt = np.linalg.svd(np.dot(landmarks_target.T, landmarks_img))
    R = (U * Vt).T

    # 得到仿射变换矩阵
    matrix = np.vstack([
        np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)),
        np.matrix([0., 0., 1.])
    ])

    # 根据shape大小创建新图，并应用仿射变换矩阵
    result = np.zeros(shape_target, dtype=img.dtype)
    cv2.warpAffine(img,
                   matrix[:2], (shape_target[1], shape_target[0]),
                   dst=result,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return result


def face_morph(bottom_img: np.ndarray,
               morph_img: np.ndarray,
               alpha: float = 0.5,
               exposure_alpha: float = 1.1) -> np.ndarray:
    """人脸融合

    传入底图、融合图和融合系数，系数越低越趋近原图，系数越高越趋近融合图

    Args:
        bottom_img (np.ndarray): 底图
        morph_img (np.ndarray): 融合图
        alpha (float, optional): 融合系数，默认值为0.5
        exposure_alpha (float, optional): 曝光系数，当出现曝光过度的蓝点时
            调高此系数. Defaults to 1.1.

    Returns:
        np.ndarray: 融合后图片
    """
    # 获得68个人脸关键点的坐标
    landmarks_bottom = get_face_68_landmarks(bottom_img)

    # 将融合图应用仿射变换并调整大小，使其和底图脸部部位大致相同
    morph_img = affine_transform_and_change_shape(morph_img, landmarks_bottom,
                                                  bottom_img.shape)

    # 定位融合图特征点
    landmarks_mask = get_face_68_landmarks(morph_img)

    # 三角融合：三角剖分，线性加权
    morph_img = triangle_face_morph(bottom_img, morph_img, landmarks_bottom,
                                    landmarks_mask, float(alpha))

    # 泊松融合
    merged_img = merge_img(bottom_img,
                           morph_img,
                           landmarks_bottom,
                           exposure_alpha=exposure_alpha)

    # 融合后存在模糊, 进行锐化并平滑
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    merged_img = cv2.filter2D(merged_img, -1, kernel=kernel)  # 锐化
    merged_img = cv2.blur(merged_img, (3, 1))  # 平滑

    return merged_img


if __name__ == "__main__":

    customer_img = cv2.imread(customer_img_path)
    addict_img = cv2.imread(addict_img_path)

    assert customer_img is not None, 'empty image'
    assert addict_img is not None, 'empty image'

    now = time.time()
    if cut_image:
        customer_img = detect_face_and_cut(customer_img)

    # 调用瘦脸算法
    customer_thin_image = face_thin(customer_img, alpha=face_thin_alpha)
    customer_thin_image_2 = face_thin(customer_thin_image,
                                      alpha=face_thin_alpha)

    # 人脸融合
    merge_image = face_morph(customer_thin_image,
                             addict_img,
                             alpha=0.2,
                             exposure_alpha=exposure_alpha)
    merge_image_2 = face_morph(customer_thin_image,
                               addict_img,
                               alpha=0.4,
                               exposure_alpha=exposure_alpha)
    merge_image_3 = face_morph(customer_thin_image_2,
                               addict_img,
                               alpha=0.6,
                               exposure_alpha=exposure_alpha)
    merge_image_4 = face_morph(customer_thin_image_2,
                               addict_img,
                               alpha=0.8,
                               exposure_alpha=exposure_alpha)

    print(f'time usage:{time.time()-now}')

    # 保存输出图像
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cv2.imwrite(f"{output_path}/customer.png", customer_img)
    cv2.imwrite(f"{output_path}/addict.png", addict_img)
    cv2.imwrite(f"{output_path}/merge.png", merge_image)
    cv2.imwrite(f"{output_path}/merge2.png", merge_image_2)
    cv2.imwrite(f"{output_path}/merge3.png", merge_image_3)
    cv2.imwrite(f"{output_path}/merge4.png", merge_image_4)

    print(f'Successfully write result to path: `{output_path}`!')
