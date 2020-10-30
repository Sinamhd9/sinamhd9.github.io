```
# !pip install opencv-contrib-python==4.4.0.44

import cv2
print(cv2.__version__)
import sys
import numpy as np


def ex_find_homography_ransac(list_pairs_matched_keypoints, threshold_ratio_inliers=0.85, threshold_reprojtion_error=3,
                              max_num_trial=1000):
    '''
    Apply RANSAC algorithm to find a homography transformation matrix that align 2 sets of feature points, transform the first set of feature point to the second (e.g. warp image 1 to image 2)
    :param list_pairs_matched_keypoints: has the format as a list of pairs of matched points: [[[p1x,p1y],[p2x,p2y]],....]
    :param threshold_ratio_inliers: threshold on the ratio of inliers over the total number of samples, accept the estimated homography if ratio is higher than the threshold
    :param threshold_reprojtion_error: threshold of reprojection error (measured as euclidean distance, in pixels) to determine whether a sample is inlier or outlier
    :param max_num_trial: the maximum number of trials to do take sample and do testing to find the best homography matrix
    :return best_H: the best found homography matrix
    '''
    l = list_pairs_matched_keypoints
    print('length(list_pairs_matched_keypoints) =', len(l))
    u = np.zeros((len(l), 1))
    v = np.zeros((len(l), 1))
    up = np.zeros((len(l), 1))
    vp = np.zeros((len(l), 1))
    for i in range(len(l)):
        u[i] = l[i][0][0]
        v[i] = l[i][0][1]
        up[i] = l[i][1][0]
        vp[i] = l[i][1][1]
    num_trial = 0
    inlrNum = 0
    while num_trial < max_num_trial and inlrNum < int(threshold_ratio_inliers * len(l)):
        s = np.random.randint(len(u), size=(4, 1))
        A = np.zeros((0, 9))
        for i in range(len(s)):
            A = np.append(A,
                          np.array([[0, 0, 0, -u[s[i]], -v[s[i]], -1, vp[s[i]] * u[s[i]], vp[s[i]] * v[s[i]], vp[s[i]]],
                                    [u[s[i]], v[s[i]], 1, 0, 0, 0, -up[s[i]] * u[s[i]], -up[s[i]] * v[s[i]],
                                     -up[s[i]]]]).astype(int), axis=0)
            uu, ss, vh = np.linalg.svd(A, full_matrices=True)
            H = np.reshape(vh[-1, :], [3, 3])
        inlrNum = 0
        inlrs = []
        for (point1, point2) in l:
            p1H = np.reshape(np.append(point1, [1]), (3, 1))
            p1pred = np.matmul(H, p1H)
            p1pred = np.reshape(p1pred[:2] / p1pred[-1], (1, 2))
            p2H = np.reshape(np.append(point2, [1]), (3, 1))
            p2pred = np.matmul(np.linalg.inv(H), p2H)
            p2pred = np.reshape(p2pred[:2] / p2pred[-1], (1, 2))
            er = np.linalg.norm(point2 - p1pred) + np.linalg.norm(point1 - p2pred)
            if er < threshold_reprojtion_error:
                inlrNum += 1
                inlrs.append([point1, point2])
        num_trial += 1
    print('Number of inliers = ', inlrNum)
    A_best = np.zeros((0, 9))
    for inlr in inlrs:
        A_best = np.append(A_best, np.array(
            [[0, 0, 0, -inlr[0][0], -inlr[0][1], -1, inlr[1][1] * inlr[0][0], inlr[1][1] * inlr[0][1], inlr[1][1]],
             [inlr[0][0], inlr[0][1], 1, 0, 0, 0, -inlr[1][0] * inlr[0][0], -inlr[1][0] * inlr[0][1], -inlr[1][0]]]),
                           axis=0)
    uu, ss, vh = np.linalg.svd(A_best, full_matrices=True)
    best_H = np.reshape(vh[-1, :], [3, 3])
    print("Best H = ", best_H)
    return best_H


```

    4.4.0
    


```

def ex_extract_and_match_feature(img_1, img_2, ratio_robustness=0.7):
    '''
    1/ extract SIFT feature from image 1 and image 2,
    2/ use a bruteforce search to find pairs of matched features: for each feature point in img_1, find its best matched feature point in img_2
    3/ apply ratio test to select the set of robust matched points
    :param img_1: input image 1
    :param img_2: input image 2
    :param ratio_robustness: ratio for the robustness test
    :return list_pairs_matched_keypoints: has the format as list of pairs of matched points: [[[p1x,p1y],[p2x,p2y]]]
    '''
    sift = cv2.xfeatures2d.SIFT_create()
    kp_img1, des_img1 = sift.detectAndCompute(cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY), None)
    kp_img2, des_img2 = sift.detectAndCompute(cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY), None)
    dis = np.zeros((len(des_img1), len(des_img2)))
    for i in range(len(des_img1)):
        for j in range(len(des_img2)):
            dis[i, j] = np.linalg.norm(des_img1[i] - des_img2[j])
    res = np.zeros((len(dis), 2))
    for i in range(len(dis)):
        d1 = np.min(dis[i, :])
        d1_arg = np.argmin(dis[i, :])
        d2 = np.min(np.array(dis[i, :])[dis[i, :] != np.min(dis[i, :])])
        if d1 / d2 < ratio_robustness:
            res[i] = [i, d1_arg]
    features = (res[~(res == 0).all(1)])
    list_pairs_matched_keypoints = []
    p = np.zeros((len(features), 2))
    p_p = np.zeros((len(features), 2))
    for i in range(len(features)):
        p[i] = kp_img1[int(features[i, 0])].pt
        p_p[i] = kp_img2[int(features[i, 1])].pt
        list_pairs_matched_keypoints.append([p[i], p_p[i]])
    print('list_pairs_matched_keypoints = ', list_pairs_matched_keypoints)
    return list_pairs_matched_keypoints
```


```
def ex_warp_blend_crop_image(img_1, H_1, img_2):
    '''
    1/ warp image img_1 using the homography H_1 to align it with image img_2 (using backward warping and bilinear resampling)
    2/ stitch image img_1 to image img_2 and apply average blending to blend the 2 images into a single panorama image
    3/ find the best bounding box for the resulting stitched image
    :param img_1:
    :param H_1:
    :param img_2:
    :return img_panorama: resulting panorama image
    '''
    dim = np.shape(img_2)
    h = dim[0]
    w = dim[1]
    canv = np.zeros((3 * dim[0], 3 * dim[1], 3))
    mask = np.zeros((3 * dim[0], 3 * dim[1], 3))
    c00 = np.matmul(H_1, np.array([[0], [0], [1]]))
    c11 = np.matmul(H_1, np.array([[w], [h], [1]]))
    c10 = np.matmul(H_1, np.array([[w], [0], [1]]))
    c01 = np.matmul(H_1, np.array([[0], [h], [1]]))
    c00 = c00[:2] / c00[2]
    c11 = c11[:2] / c11[2]
    c10 = c10[:2] / c10[2]
    c01 = c01[:2] / c01[2]
    left = int(np.floor(np.minimum(c00[0], c01[0])))
    right = int(np.maximum(np.ceil(np.maximum(c10[0], c11[0])), w))
    up = int(np.floor(np.minimum(c00[1], c10[1])))
    down = int(np.maximum(np.ceil(np.maximum(c01[1], c11[1])), h))
    for i in range(-400, 400):
        for j in range(-400, 400):
            z = np.matmul(np.linalg.inv(H_1), np.array([[j], [i], [1]]))
            z = z[:2] / z[2]
            a = z[0] - np.floor(z[0])
            b = z[1] - np.floor(z[1])
            z = np.floor(z).astype(int)
            if h - 1 > z[1] >= 0 and w - 1 > z[0] >= 0:     # bilinear interpolation
                canv[i + h, j + w, :] = (1 - a) * (1 - b) * img_1[z[1], z[0], :] + b * (1 - a) * img_1[z[1] + 1, z[0],
                                                                                                 :] + a * b * img_1[
                                                                                                              z[1] + 1,
                                                                                                              z[0] + 1,
                                                                                                              :] + (
                                                1 - b) * a * img_1[z[1], z[0] + 1, :] + canv[i + h, j + w, :]
                mask[i + h, j + w] += 1
            if h > i >= 0 and w > j >= 0:
                canv[i + h, j + w] += img_2[i, j]
                mask[i + h, j + w] += 1

    mask[mask == 0] = 1
    canv /= mask
    img_panorama = canv[h + up - 1:h + down + 1, w + left - 1:w + right + 1]  # Crop
    return img_panorama

```


```
def stitch_images(img_1, img_2):
    '''
    :param img_1: input image 1. We warp this image to align and stich it to the image 2
    :param img_2: is the reference image. We will not warp this image
    :return img_panorama: the resulting stiched image
    '''
    print('==============================')
    print('===== stitch two images to generate one panorama image')
    print('==============================')
    # ===== extract and match features from image 1 and image 2
    list_pairs_matched_keypoints = ex_extract_and_match_feature(img_1=img_1, img_2=img_2, ratio_robustness=0.7)

    # ===== use RANSAC algorithm to find homography to warp image 1 to align it to image 2
    H_1 = ex_find_homography_ransac(list_pairs_matched_keypoints, threshold_ratio_inliers=0.85,
                                    threshold_reprojtion_error=3, max_num_trial=1000)

    # ===== warp image 1, blend it with image 2 using average blending to produce the resulting panorama image
    img_panorama = ex_warp_blend_crop_image(img_1=img_1, H_1=H_1, img_2=img_2)

    return img_panorama
```


```
from google.colab.patches import cv2_imshow
# Visualization
img_1 = cv2.imread('im3.jpg')
img_2 = cv2.imread('im4.jpg')
img_panorama = stitch_images(img_1=img_1, img_2=img_2)
cv2.imwrite('stiched.png', img_panorama)
cv2_imshow(img_panorama)
```

    ==============================
    ===== stitch two images to generate one panorama image
    ==============================
    list_pairs_matched_keypoints =  [[array([39.25055313, 14.85711193]), array([109.62735748,  14.03421402])], [array([69.25901794, 52.17156219]), array([ 7.12865734, 46.79865265])], [array([72.50183868, 53.5447731 ]), array([11.17586899, 48.43230438])], [array([74.39653778, 49.94051743]), array([13.78067207, 44.40942383])], [array([83.65226746, 50.03367996]), array([24.7720356 , 45.83179092])], [array([83.65226746, 50.03367996]), array([24.7720356 , 45.83179092])], [array([83.71414948, 50.00365829]), array([24.7720356 , 45.83179092])], [array([86.64093781, 52.12395477]), array([28.60063553, 47.97027969])], [array([87.07094574, 59.02228546]), array([28.77672005, 55.65923691])], [array([89.52572632, 42.55262756]), array([32.51744843, 38.13439178])], [array([94.11037445, 46.81702805]), array([37.75405502, 43.69388962])], [array([97.79826355, 25.18143463]), array([43.0647583 , 21.31850243])], [array([98.54994202, 33.69439697]), array([43.21373749, 29.32772064])], [array([99.0022583 , 44.57745743]), array([42.82207108, 41.38788605])], [array([100.04170227,  50.89116669]), array([43.60903931, 48.07452774])], [array([100.11974335,  64.7464447 ]), array([44.12991714, 62.55490494])], [array([102.02485657, 133.28327942]), array([ 41.80261993, 135.70643616])], [array([103.31407928,  28.38159561]), array([48.47322845, 24.53434944])], [array([104.24066162,  53.93753815]), array([48.23449707, 50.97293091])], [array([106.2258606 ,  38.88117218]), array([51.31092834, 35.79350662])], [array([108.36761475,  52.24137115]), array([53.43862152, 49.95211792])], [array([108.38523865,  48.47771072]), array([53.44960785, 45.90469742])], [array([108.50709534, 129.05535889]), array([ 48.54491425, 129.99589539])], [array([111.99771118,  65.93787384]), array([57.14877319, 64.17816162])], [array([112.65161133,  56.2542038 ]), array([57.76997375, 54.28363037])], [array([115.32365417,  50.79650879]), array([60.54444504, 48.73796082])], [array([116.36549377,  47.80870819]), array([62.06155777, 46.08042908])], [array([116.52340698,  42.78087616]), array([62.4559021, 40.8740654])], [array([116.76413727, 133.32577515]), array([ 57.32730103, 133.03564453])], [array([120.84725952,  67.10700226]), array([66.30278778, 66.09978485])], [array([125.60642242,  57.68572235]), array([71.56324768, 56.81813431])], [array([134.19599915,  68.44355774]), array([79.63650513, 68.32228851])], [array([135.22091675,  42.14717865]), array([80.95844269, 41.63954926])], [array([135.44042969,   9.86027718]), array([81.59736633, 10.09725475])], [array([136.02825928,  65.2537384 ]), array([82.07685089, 64.98679352])], [array([138.39195251,  37.61931229]), array([84.32813263, 38.11023331])], [array([138.97045898,   8.47959232]), array([85.30621338,  9.31110668])], [array([138.97045898,   8.47959232]), array([85.30621338,  9.31110668])], [array([140.00527954,  49.09929276]), array([85.38824463, 49.57976532])], [array([140.89393616,  25.53308487]), array([86.78623199, 26.26095772])], [array([140.89393616,  25.53308487]), array([86.78623199, 26.26095772])], [array([141.4539032 ,  41.22337341]), array([87.06882477, 41.90707016])], [array([142.02485657,  67.95774841]), array([87.08159637, 67.80702972])], [array([142.4072113 ,  21.50521469]), array([87.7399292 , 22.59231186])], [array([142.4072113 ,  21.50521469]), array([87.7399292 , 22.59231186])], [array([143.55833435,  46.11293411]), array([88.83933258, 46.60967636])], [array([145.70303345,  67.0042572 ]), array([90.36331177, 66.97875214])], [array([146.19012451,  59.16914749]), array([91.04270172, 59.50078583])], [array([147.29240417,   5.63500786]), array([92.98138428,  7.2130332 ])], [array([149.26040649,  70.54360199]), array([93.32488251, 70.66181946])], [array([149.36930847,  53.21205139]), array([94.09633636, 53.96846771])], [array([151.41662598,  68.69072723]), array([95.44908905, 68.90348816])], [array([152.3611908 ,  79.60899353]), array([95.77832794, 79.10203552])], [array([156.42460632,  26.97292519]), array([100.9437561 ,  29.67440605])], [array([156.78730774, 118.68845367]), array([ 98.71404266, 115.84568787])], [array([159.54391479,  60.89403534]), array([102.34790039,  61.73659897])], [array([162.11941528,  74.00721741]), array([104.50056458,  74.06887817])], [array([163.85406494,  35.33593369]), array([107.60527802,  39.89380264])], [array([164.75474548,  14.36315727]), array([107.25655365,  11.32952595])], [array([165.20111084,  31.67102242]), array([108.21479034,  34.84095001])], [array([166.01634216,   9.2830658 ]), array([109.62735748,  14.03421402])], [array([167.12666321,  22.33870697]), array([110.67934418,  25.53856659])], [array([167.57662964,  21.36957741]), array([110.67934418,  25.53856659])], [array([167.73179626,  56.86955643]), array([109.82284546,  58.60693741])], [array([167.73179626,  56.86955643]), array([109.82284546,  58.60693741])], [array([168.34721375,  25.84514427]), array([111.00136566,  29.78504181])], [array([169.8745575 ,  78.80638123]), array([111.07753754,  78.7857132 ])], [array([172.05131531,  36.41457367]), array([114.10209656,  40.07992172])], [array([174.22686768,  65.92061615]), array([114.95381927,  67.0230484 ])], [array([180.39537048,  53.85344696]), array([120.13375854,  56.81858444])], [array([180.39537048,  53.85344696]), array([120.13375854,  56.81858444])], [array([181.62106323, 113.77922821]), array([119.42801666, 110.5253067 ])], [array([182.60699463,  35.24960327]), array([122.81575012,  39.57168198])], [array([183.06387329,  76.79248047]), array([121.61555481,  77.0396347 ])], [array([186.0103302 ,  50.62780762]), array([125.13192749,  54.22072601])], [array([186.0103302 ,  50.62780762]), array([125.13192749,  54.22072601])], [array([194.69520569,  59.30220032]), array([131.92237854,  61.44125366])]]
    length(list_pairs_matched_keypoints) = 77
    Number of inliers =  72
    Best H =  [[-1.52691350e-02  5.20252640e-04  9.49343606e-01]
     [-2.37500813e-03 -1.32970460e-02  3.13424378e-01]
     [-2.76703714e-05  6.05759405e-07 -9.79828335e-03]]
    


![png](image_stitch_files/image_stitch_4_1.png)



```

```
