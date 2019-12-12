# -*- coding:utf-8 -*-

import keras

def outer_product(x):

    return keras.backend.batch_dot(
                x[0]
                , x[1]
                , axes=[1,1]
            ) / x[0].get_shape().as_list()[1] 

def signed_sqrt(x):

    return keras.backend.sign(x) * keras.backend.sqrt(keras.backend.abs(x) + 1e-9)

def L2_norm(x, axis=-1):

    return keras.backend.l2_normalize(x, axis=axis)

def bilinear_pooling(x):

    """
    实现双线性池化，双线性池化出要使用在细粒度图像分类中
    具体参考论文：Bilinear CNN Models for Fine-grained Visual Recognition
    论文链接：http://vis-www.cs.umass.edu/bcnn/docs/bcnn_iccv15.pdf
    传入x:普通分类网络的最后一个卷积层的输出
    如果想要放置在池化层之后需要进行适当的修改
    可以直接插入，最后一个卷积层与全连接层之前
    """
    shape_detector = x.shape
    print("shape_detector : {}".format(shape_detector))

    # extract features from extractor , same with detector for symmetry DxD model
    shape_extractor = shape_detector
    x_extractor = x_detector = x
    print("shape_extractor : {}".format(shape_extractor))
    # print(shape_extractor[0],shape_detector[1],shape_detector[2])
        
    
    # rehape to (minibatch_size, total_pixels, filter_size)
    x_detector = keras.layers.Reshape(
            [
                1 * 1 , shape_detector[-1]
            ]
        )(x_detector)
    print("x_detector shape after rehsape ops : {}".format(x_detector.shape))
        
    x_extractor = keras.layers.Reshape(
            [
                1 * 1 , shape_extractor[-1]
            ]
        )(x_extractor)
    print("x_extractor shape after rehsape ops : {}".format(x_extractor.shape))
        
        
    # outer products of features, output shape=(minibatch_size, filter_size_detector*filter_size_extractor)
    x = keras.layers.Lambda(outer_product)(
        [x_detector, x_extractor]
    )
    print("x shape after outer products ops : {}".format(x.shape))
        
        
    # rehape to (minibatch_size, filter_size_detector*filter_size_extractor)
    x = keras.layers.Reshape([shape_detector[-1]*shape_extractor[-1]])(x)
    print("x shape after rehsape ops : {}".format(x.shape))
        
        
    # signed square-root 
    x = keras.layers.Lambda(signed_sqrt)(x)
    print("x shape after signed-square-root ops : {}".format(x.shape))
        
    # L2 normalization
    x = keras.layers.Lambda(L2_norm)(x)
    print("x shape after L2-Normalization ops : {}".format(x.shape))

    return x

