# Original Version: Taehoon Kim (http://carpedm20.github.io)
#   + Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/model.py
#   + License: MIT
# [2016-08-05] Modifications for Completion: Brandon Amos (http://bamos.github.io)
#   + License: MIT


# LBN  2018年4月16日 10:34:00
#  纯净 DCGAN

from __future__ import division
import os
import time
import math
import itertools
from glob import glob
import cv2 as cv
from six.moves import range

from ops import *
from utils import *

SUPPORTED_EXTENSIONS = ["png", "jpg", "jpeg"]


# 定义路径，读入文件夹下的图片
def dataset_files(root):
    """Returns a list of all image files in the given directory"""
    return list(itertools.chain.from_iterable(
        glob(os.path.join(root, "*.{}".format(ext))) for ext in SUPPORTED_EXTENSIONS))


class myDCGAN_7(object):
    def __init__(self, sess, image_size=64, is_crop=False,
                 batch_size=64, sample_size=64, lowres=8,
                 z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3,
                 checkpoint_dir=None, lam=0.1):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            lowres: (optional) Low resolution image/mask shrink factor. [8]     低分辨率图像/蒙版的收缩系数
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen untis for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. [3]
        """
        # Currently, image size must be a (power of 2) and (8 or higher).
        assert (image_size & (image_size - 1) == 0 and image_size >= 8)

        self.sess = sess
        self.is_crop = is_crop
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.image_shape = [image_size, image_size, c_dim]  # 64*64*3

        self.lowres = lowres
        self.lowres_size = image_size // lowres
        self.lowres_shape = [self.lowres_size, self.lowres_size, c_dim]

        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.lam = lam

        self.c_dim = c_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bns = [
            batch_norm(name='d_bn{}'.format(i, )) for i in range(4)]

        log_size = int(math.log(image_size) / math.log(2))
        self.g_bns = [
            batch_norm(name='g_bn{}'.format(i, )) for i in range(log_size)]

        self.checkpoint_dir = checkpoint_dir  # 模型保存
        self.build_model()  # 建立模型

        self.model_name = "DCGAN.model"

#____________________________________________________________________________________________________________________________
    # 建立语义模型，和相关参数
    def build_model(self):
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        #设定输入批图像格式： [None] + self.image_shape
        self.images = tf.placeholder(
            tf.float32, [None] + self.image_shape, name='real_images')


        #z：鉴别网络计算值，放入tensorboard
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.z_sum = tf.summary.histogram("z", self.z)

        #生成网络模型
        self.G = self.generator(self.z)

        # 2个判别网络模型
        self.D_img, self.D_global = self.discriminator(self.images)

        self.D_gen, self.D_local = self.discriminator(self.G, reuse=True)

        #计入tensorboard
        self.d_sum = tf.summary.histogram("d", self.D_img)
        self.d__sum = tf.summary.histogram("d_", self.D_gen)
        self.G_sum = tf.summary.image("G", self.G)

        #制定1个生成损失，2个鉴别损失，都是用来训练discriminator
        #sigmoid_cross_entropy_with_logits： 函数的作用是计算经sigmoid 函数激活之后的交叉熵。
        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_global,
                                                    labels=tf.ones_like(self.D_img)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_local,
                                                    labels=tf.zeros_like(self.D_gen)))
        #g_loss 生成损失
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_local,
                                                    labels=tf.ones_like(self.D_gen)))
        #计入tensorboard
        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        #d_loss 判别损失
        self.d_loss = self.d_loss_real + self.d_loss_fake

        #计入tensorboard
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(max_to_keep=500)




#-----------------------------------------------------------------------------------------------------------------------------------

    # 定义训练和参数
    def train(self, config):

        # 读入数据
        data = dataset_files(config.dataset)
        np.random.shuffle(data)  # 让训练数据集中的数据打乱顺序
        assert (len(data) > 0)

        # 鉴别网络和生成网络优化
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        # 计入tensorboard
        self.g_sum = tf.summary.merge(
            [self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge(
            [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        # 计入tensorboard，写入文件名：logs
        self.writer = tf.summary.FileWriter("./logs7", self.sess.graph)

        #
        sample_z = np.random.uniform(-1, 1, size=(self.sample_size, self.z_dim))  # （64，100）

        # 把64个图像文件转换成标准数据集需求的数组文件（64，64，64，3），64张64*64维彩色图像
        sample_files = data[0:self.sample_size]  # 64个图片文件
        sample = [get_image(sample_file, self.image_size, is_crop=self.is_crop) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)

        counter = 1
        start_time = time.time()

        # mask
        if config.maskType == 'center':
            assert (config.centerScale <= 0.5)  # 中心上下左右各掩盖不超过50%
            Scale = 0.5 - config.centerScale / 2

            # 破损区域的 bro_mask
            bro_mask = np.zeros(self.image_shape)  # 初始化mask全为0，形状：64*64*3
            sz = self.image_size  # image_size = 64
            l = int(self.image_size * Scale)  # l = 64*0.25 = 16
            u = int(self.image_size * (1.0 - Scale))  # u = 64*(1-0.25) = 48
            bro_mask[l:u, l:u, :] = 1.0  # mask【L:U】为1，表示破损区域选定

            # 完好区域的ok_mask
            ok_mask = np.ones(self.image_shape)  # 初始化mask全为1，形状：64*64*3
            ok_mask = ok_mask - bro_mask  # mask【L:U】为1，表示完好区域选定


        # 规定了训练的回合数config.epoch = flags（epoch）
        for epoch in range(config.epoch):
            # 读取数据
            data = dataset_files(config.dataset)
            np.random.shuffle(data)  # 让训练数据集中的数据打乱顺序
            assert (len(data) > 0)

            # 取余，config.train_size：正无穷大
            batch_idxs = min(len(data), config.train_size) // self.batch_size

            # 在训练集中，分批次训练，批次大小为batch_size=64
            for idx in range(0, batch_idxs):

                # 在数据集中选64个图像，并转成batch_images:（64，64，64，3）
                batch_files = data[idx * config.batch_size:(idx + 1) * config.batch_size]
                batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop)
                         for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)

                # batch_z = (64,100)
                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                    .astype(np.float32)

                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                                               feed_dict={self.images: batch_images, self.z: batch_z,
                                                          self.is_training: True})
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={self.z: batch_z, self.is_training: True})
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={self.z: batch_z, self.is_training: True})
                self.writer.add_summary(summary_str, counter)

                # 计算3种损失：d_loss_fake；d_loss_real；g_loss
                errD_fake = self.d_loss_fake.eval({self.z: batch_z, self.is_training: False})
                errD_real = self.d_loss_real.eval({self.images: batch_images, self.is_training: False})
                errG = self.g_loss.eval({self.z: batch_z, self.is_training: False})

                counter += 1
                print(
                    "Epoch: [{:2d}] [{:4d}/{:4d}] time: {:4.4f}, d_loss: {:.8f}（fake:{:.8f},real:{:.8f}）, g_loss: {:.8f}".format(
                        epoch, idx, batch_idxs, time.time() - start_time, errD_fake + errD_real, errD_fake, errD_real,
                        errG))

                # 取余，每100次迭代存一次图片，每500次迭代存一次模型
                if np.mod(counter, 150) == 1:
                    samples, d_loss, g_loss = self.sess.run(
                        [self.G, self.d_loss, self.g_loss],
                        feed_dict={self.z: sample_z, self.images: sample_images, self.is_training: False}
                    )

                    completed_images = np.multiply(samples,bro_mask)+np.multiply(sample_images,ok_mask)
                    save_images(completed_images, [8, 8],
                                './samples/train_{:02d}_{:04d}.png'.format(epoch, idx))
                    print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))

                    self.save(config.checkpoint_dir, counter)





    #--------------NET Worker-------------------------------------------------------------------------------------------------------------------
    def discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            # TODO: Investigate how to parameterise discriminator based off image size.
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bns[0](conv2d(h0, self.df_dim * 2, name='d_h1_conv'), self.is_training))
            h2 = lrelu(self.d_bns[1](conv2d(h1, self.df_dim * 4, name='d_h2_conv'), self.is_training))
            h3 = lrelu(self.d_bns[2](conv2d(h2, self.df_dim * 8, name='d_h3_conv'), self.is_training))
            h4 = linear(tf.reshape(h3, [-1, 8192]), 1, 'd_h4_lin')

            return tf.nn.sigmoid(h4), h4


    def generator(self, z):
        with tf.variable_scope("generator") as scope:
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim * 8 * 4 * 4, 'g_h0_lin', with_w=True)

            # TODO: Nicer iteration pattern here. #readability
            hs = [None]
            hs[0] = tf.reshape(self.z_, [-1, 4, 4, self.gf_dim * 8])
            hs[0] = tf.nn.relu(self.g_bns[0](hs[0], self.is_training))

            i = 1  # Iteration number.
            depth_mul = 8  # Depth decreases as spatial component increases.
            size = 8  # Size increases as depth decreases.

            while size < self.image_size:
                hs.append(None)
                name = 'g_h{}'.format(i)
                hs[i], _, _ = conv2d_transpose(hs[i - 1],
                                               [self.batch_size, size, size, self.gf_dim * depth_mul], name=name,
                                               with_w=True)
                hs[i] = tf.nn.relu(self.g_bns[i](hs[i], self.is_training))

                i += 1
                depth_mul //= 2
                size *= 2

            hs.append(None)
            name = 'g_h{}'.format(i)
            hs[i], _, _ = conv2d_transpose(hs[i - 1],
                                           [self.batch_size, size, size, 3], name=name, with_w=True)

            return tf.nn.tanh(hs[i])


    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, self.model_name),
                        global_step=step)