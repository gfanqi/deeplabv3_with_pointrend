import os
import shutil
from functools import partial
from random import random

import cv2
import numpy as np
import tensorflow as tf
from matplotlib.backends.backend_agg import FigureCanvasAgg
from tensorflow.python.keras.optimizers import Adam

from core.deeplabV3plus import Deeplabv3
from core.pointrend import PointRend


def preprocessing(image_path, H=512, W=512):
    x = tf.io.read_file(image_path)
    x = tf.cond(
        tf.image.is_jpeg(x),
        lambda: tf.image.decode_jpeg(x, channels=3),
        lambda: tf.image.decode_png(x, channels=3))

    x = tf.cast(x, tf.float32)
    x = x / 255.
    x = tf.image.resize(x, size=(H, W))
    return x


# def plot(pred):
#     fig = plt.figure()
#     # 创建3d图形的两种方式
#     # ax = Axes3D(fig)
#     H, W = pred.shape
#     ax = fig.add_subplot(111, projection='3d')
#     # X, Y value
#     X = np.arange(0, W, 1)
#     Y = np.arange(0, H, 1)
#     X, Y = np.meshgrid(X, Y)  # x-y 平面的网格
#     # R = np.sqrt(X ** 2 + Y ** 2)
#     # height value
#     # Z = np.sin(R)
#     Z = pred
#     # rstride:行之间的跨度  cstride:列之间的跨度
#     # rcount:设置间隔个数，默认50个，ccount:列的间隔个数  不能与上面两个参数同时出现
#     # vmax和vmin  颜色的最大值和最小值
#     ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
#     # zdir : 'z' | 'x' | 'y' 表示把等高线图投射到哪个面
#     # offset : 表示等高线图投射到指定页面的某个刻度
#     ax.contourf(X, Y, Z, zdir='z', offset=-2)
#     # 设置图像z轴的显示范围，x、y轴设置方式相同
#     ax.set_zlim(0, 2)
#     # plt.show()
#     plt.savefig('temp.png')
def plot(pred, randname=None):
    H, W = pred.shape
    fig = plt.figure('quiver矢量化的图片')
    X = np.arange(0, W, 1)
    Y = np.arange(0, H, 1)
    contour = plt.contour(X, Y, pred[::-1, ...], [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1])
    plt.clabel(contour, fontsize=10, )
    plt.axis('off')
    fig.set_size_inches(W / 100, H / 100)  # 输出width*height像素
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    canvas = FigureCanvasAgg(plt.gcf())
    canvas.draw()
    img = np.array(canvas.renderer.buffer_rgba())
    plt.close()
    return img


def choose_confidence(array):
    num_list = []
    for conf in np.linspace(0.1, 1.2, num=30):
        # conf = confidence * 0.1
        nums = np.sum(array > conf) / np.sum(array > 0.1)
        num_list.append(nums)

    grad = [(num_list[i] - num_list[i + 1]) for i in range(len(num_list) - 1)]
    grad_df = pd.DataFrame({
        'confidence': np.linspace(0.1, 1.2, num=29, endpoint=False),
        'grad': grad
    })
    selected_grad_df = grad_df[grad_df['grad'] < 0.02]
    if len(selected_grad_df) <= 1:
        confidence = 0.5
    else:
        confidence =(np.max(selected_grad_df['confidence'])+np.median(selected_grad_df['confidence']))/2
    return confidence, np.mean(grad_df['grad']) * 100, len(selected_grad_df) / len(grad_df)


def res_func(res, path_batch, class_name):
    if not os.path.exists(os.path.join(save_dir, class_name)): os.makedirs(os.path.join(save_dir, class_name))
    for index, (single_res, image_path) in list(enumerate(zip(res, path_batch))):
        image_path = image_path.decode()
        ori_image = cv2.imread(image_path)
        crop_image = ori_image.copy()
        H, W, c = ori_image.shape
        pred = cv2.resize(single_res, dsize=(W, H))
        rand_name = int(random() * 100000000000)
        # pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
        plot_image = plot(pred, rand_name)
        # plot_image = cv2.imread('./images/{}.png'.format(rand_name))
        plot_image = cv2.resize(plot_image, dsize=(W, H))
        plot_image = cv2.cvtColor(plot_image, cv2.COLOR_RGBA2BGR)
        plot_image_gray = cv2.cvtColor(plot_image, cv2.COLOR_BGR2GRAY)
        confidence, metric1, metric2 = choose_confidence(pred)
        crop_image[pred < confidence] = np.array([0, 0, 0])
        plot_image[plot_image_gray == 255] = ori_image[plot_image_gray == 255]
        plot_image[plot_image_gray != 255] = ori_image[plot_image_gray != 255] * 0.1 + plot_image[
            plot_image_gray != 255] * 0.9
        image = np.concatenate([ori_image, crop_image, plot_image], axis=-2)
        image = image.astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, class_name,
                                 '{}_{}_{}-'.format(int(metric1*1000000000), int(metric2 * 10000000),int(confidence*100)) + os.path.basename(image_path)),
                    image)
        print('\rdone!', end='')

    # print('done')


def res_func_test(res, path_batch, class_name):
    for index, (single_res, image_path) in list(enumerate(zip(res, path_batch))):
        image_path = image_path.decode()
        ori_image = cv2.imread(image_path)
        crop_image = ori_image.copy()
        H, W, c = ori_image.shape
        pred = cv2.resize(single_res, dsize=(W, H))
        confidence, metric1, metric2 = choose_confidence(pred)
        if not os.path.exists(os.path.join(save_dir, 'confidence', '{}'.format(confidence))): os.makedirs(
            os.path.join(save_dir, 'confidence', '{}'.format(confidence)))
        crop_image[pred < confidence] = np.array([0, 0, 0])
        image = crop_image.astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, 'confidence', '{}'.format(confidence),
                                 '{}_{}'.format(int(metric1 * 1000), int(metric2 * 1000)) + os.path.basename(
                                     image_path)),
                    image)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    test_pointrend = True
    use_multiprocess = False
    test_mode = True

    save_dir = os.path.expanduser('~/Image/data/temp_segmentation/')
    if test_mode:
        if os.path.exists(save_dir): shutil.rmtree(save_dir)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    HEIGHT = 512
    WIDTH = 512
    batch_size_per_replica = 3
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    num_gpus = len(gpus)
    with  tf.distribute.MirroredStrategy().scope():
        if test_pointrend:
            deeplabv3plus = Deeplabv3(weights=None, input_shape=(HEIGHT, WIDTH, 3), classes=1, backbone='mobilenetv2')
            pointrend = PointRend(deeplabv3plus, semantic_segment_model_layer_names=[
                'expanded_conv_3_project_BN',
                # 'expanded_conv_4_add',
                # 'expanded_conv_5_add',
                'expanded_conv_6_project_BN',
                # 'expanded_conv_7_add',
                # 'expanded_conv_8_add',
                # 'expanded_conv_9_add',
                'expanded_conv_10_project_BN',
                # 'expanded_conv_11_add',
                # 'expanded_conv_12_add',
                'expanded_conv_13_project_BN',
                # 'expanded_conv_14_add',
                # 'expanded_conv_15_add',
                'expanded_conv_16_project_BN',
                'logits_semantic'], batch_size=batch_size_per_replica, num_class=1)
            # model = pointrend.build_model_for_train(important_ratio=0.75)
            model = pointrend.build_model_for_infer(batch_size=batch_size_per_replica,
                                                    weights_path='./logs/time-1605789064ep213-loss0.052-val_loss0.051.h5')
            model.summary()
        else:
            model = Deeplabv3(weights=None, input_shape=(HEIGHT, WIDTH, 3), classes=1, backbone='mobilenetv2')
            model.compile(loss=tf.keras.losses.categorical_crossentropy,
                          optimizer=Adam(lr=1e-4),
                          metrics=['accuracy'],
                          )
            model.load_weights('logs/time-1605789064ep213-loss0.052-val_loss0.051.h5', by_name=True)
    import pandas as pd

    df = pd.read_csv('total_data.csv')
    df['image_name'] = df['image_path'].map(lambda x: os.path.basename(x))
    df = df[np.logical_not(df['image_name'].isin(['duizhaung_2568896.jpg']))]

    if os.path.exists('/home/ubuntu/Image/data/temp_segmentation/confidence/0.0'):
        downloaded_image = os.listdir('/home/ubuntu/Image/data/temp_segmentation/confidence/0.0')
        df = df[np.logical_not(df['image_name'].isin(downloaded_image))]
    if use_multiprocess:
        from multiprocessing.pool import Pool

        pool = Pool(48)
    else:
        pool = None
    classes = list(set(df['image_kind'].tolist()))
    for cls in classes:
        new_df = df[df['image_kind'] == cls]
        if len(new_df) <= batch_size_per_replica * num_gpus: continue
        image_paths_list = new_df['image_path'].tolist()
        num_images = len(image_paths_list)
        image_paths_list = image_paths_list[:num_images]

        # print(image_paths)

        preprocessing = partial(preprocessing, H=HEIGHT, W=WIDTH)
        db_data = tf.data.Dataset.from_tensor_slices(image_paths_list).map(lambda x: preprocessing(x)).batch(
            batch_size_per_replica * num_gpus, drop_remainder=True).prefetch(-1)
        path_db = tf.data.Dataset.from_tensor_slices(image_paths_list).batch(batch_size_per_replica * num_gpus,
                                                                             drop_remainder=True).prefetch(-1)
        # print(res)
        db = tf.data.Dataset.zip((db_data, path_db))

        confidence = 0.8
        # crop = Crop_image()
        batch_index = 0
        print(cls)
        for path_and_data in iter(db):
            batch_index += 1
            num_pics = batch_size_per_replica * num_gpus * batch_index
            print('\r{}/{}'.format(num_pics, len(image_paths_list)), end='')
            # if batch_index > 10: break
            data_batch, path_batch = path_and_data
            path_batch = path_batch.numpy()
            res = model.predict(data_batch)
            if use_multiprocess:
                pool.apply_async(res_func, args=[res, path_batch, cls])
            else:
                res_func(res, path_batch, cls)
        if use_multiprocess:
            pool.close()
            pool.join()
        # break
