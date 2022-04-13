from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from collections import defaultdict
import re
from six.moves import xrange

from abc import ABC, abstractmethod
from random import shuffle

from ops import *
from utils import *


#gestIK data variables - base directory and sub dirs holding training data
gestIKdataBaseDir = 'E:\\Dropbox\\Public\\GestureIK\\frames'
baseDataDirList = ['CVEL_06101316','CVEL_06081106']#,'CVEL_06071511','CVEL_06062149']
#file holding all paths to example images and each example's class
gestIKdataFileName = os.path.join('.','gestIK', 'allImgFileNamesAndClass.txt')

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

def sigmoid_cross_entropy_with_logits(x, y):
  try:
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
  except:
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)


class DCGAN(ABC,object):
  className = 'DCGAN'
#  def __init__(self, sess, input_height=108, input_width=108, crop=True,
#         batch_size=64, sample_num = 64, output_height=64, output_width=64,
#         y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
#         gfc_dim=1024, dfc_dim=1024, clr_dim=3, dataset_name='default',
#         input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None):
  def __init__(self, sess, 
               flags, sample_num=64, y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
               gfc_dim=1024, dfc_dim=1024, sample_dir=None):
    """
    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      sample_num : (optional) Number of samples to take during training
      y_dim: (optional) Dimension of dim for y (conditional GAN). [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      --ignored : clr_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    self.sess = sess
    self.sample_num = sample_num
    self.sample_dir = sample_dir

    self.batch_size = flags.batch_size
    self.input_height = flags.input_height
    self.input_width = flags.input_width
    self.output_height = flags.output_height
    self.output_width = flags.output_width
    self.dataset_name = flags.dataset
    self.input_fname_pattern = flags.input_fname_pattern
    self.crop = flags.crop        
    self.checkpoint_dir = flags.checkpoint_dir

    self.y_dim = y_dim
    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')

    if not self.y_dim:
      self.d_bn3 = batch_norm(name='d_bn3')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')

    if not self.y_dim:
      self.g_bn3 = batch_norm(name='g_bn3')

    
    #load dataset sets clr_dim based on image data for vanilla dcgan/celebA data
    self.loadDataset()

    self.grayscale = (self.clr_dim == 1)

    self.build_model()
    
  @abstractmethod
  def loadDataset(self):
    pass
    
  #use d_ to denote variables pertaining to discriminator, g_ to mark generator variables
  def build_model(self):
    if self.y_dim:
      self.y= tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

    if self.crop:
      image_dims = [self.output_height, self.output_width, self.clr_dim]
    else:
      image_dims = [self.input_height, self.input_width, self.clr_dim]

    self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images')
    self.sample_inputs = tf.placeholder(tf.float32, [self.sample_num] + image_dims, name='sample_inputs')

    inputs = self.inputs
    sample_inputs = self.sample_inputs

    self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
    self.z_sum = histogram_summary('z', self.z)
    #if condGAN
    if self.y_dim:
      self.G = self.generator(self.z, self.y)
      self.D, self.D_logits = \
          self.discriminator(inputs, self.y, reuse=False)

      self.sampler = self.sampler(self.z, self.y)
      self.D_, self.D_logits_ = \
          self.discriminator(self.G, self.y, reuse=True)
    else:
      self.G = self.generator(self.z)
      self.D, self.D_logits = self.discriminator(inputs)

      self.sampler = self.sampler(self.z)
      self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)
    
    #summary variables used for logging
#    self.d_sum = histogram_summary('d', self.D)
#    self.d__sum = histogram_summary('d_', self.D_)
#    self.G_sum = image_summary('G', self.G)
    
    self.buildLossFuncs()
#    self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
#    self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
#    self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
#
#    self.d_loss_real_sum = scalar_summary('d_loss_real', self.d_loss_real)
#    self.d_loss_fake_sum = scalar_summary('d_loss_fake', self.d_loss_fake)
#                          
#    self.d_loss = self.d_loss_real + self.d_loss_fake

#    self.g_loss_sum = scalar_summary('g_loss', self.g_loss)
#    self.d_loss_sum = scalar_summary('d_loss', self.d_loss)

    #TODO: potentially conditionally build summary functionality, for speed concerns?
    self.buildSummaryFuncs()

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver()
    
  #build loss functions - override if desire other loss format
  def buildLossFuncs(self):
    self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
    self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
    self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

    self.d_loss_real_sum = scalar_summary('d_loss_real', self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary('d_loss_fake', self.d_loss_fake)
                          
    self.d_loss = self.d_loss_real + self.d_loss_fake
  
  def buildSummaryFuncs(self):
    #summary variables used for logging - can build conditionally if speed is an issue
    self.d_sum = histogram_summary('d', self.D)
    self.d__sum = histogram_summary('d_', self.D_)
    self.G_sum = image_summary('G', self.G)
    self.g_loss_sum = scalar_summary('g_loss', self.g_loss)
    self.d_loss_sum = scalar_summary('d_loss', self.d_loss)


  @abstractmethod
  def train(self, config):
    pass
  
  def sampleTrainImgs(self, config, epoch, idx, feedDict):
    samples, d_loss, g_loss = self.sess.run([self.sampler, self.d_loss, self.g_loss], feed_dict=feedDict)
    manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
    manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
    if (self.sample_dir != None):
      save_images(samples, [manifold_h, manifold_w], './{}/train_{:02d}_{:04d}.png'.format(self.sample_dir, epoch, idx))
    else :
      save_images(samples, [manifold_h, manifold_w], './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
    print('[Sample] d_loss: %.8f, g_loss: %.8f' % (d_loss, g_loss)) 
      
  @abstractmethod      
  def discriminator(self, image, y=None, reuse=False):
    return None

  @abstractmethod
  def generator(self, z, y=None):
    return None

  @abstractmethod
  def sampler(self, z, y=None):
    return None
    
  def dbgOut(self, methodName):
    print('Calling Method : %s of Class instance : %s with dataset : %s ' %(methodName, self.className, self.dataset_name))
    
  @property
  def model_dir(self):
    return '{}_{}_{}_{}'.format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)
      
  def save(self, checkpoint_dir, step):
    model_name = 'DCGAN.model'
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(' [*] Reading checkpoints...')
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
    #TODO (jt) verify loads most recent model
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer('(\d+)(?!.*\d)',ckpt_name)).group(0))
      print(' [*] Success to read {}'.format(ckpt_name))
      return True, counter
    else:
      print(' [*] Failed to find a checkpoint')
      return False, 0


#basic DCGAN implementation
class baseDCGan(DCGAN):
    className = 'baseDCGan'
#    def __init__(self, sess, input_height=108, input_width=108, crop=True,
#         batch_size=64, sample_num = 64, output_height=64, output_width=64,
#         z_dim=100, gf_dim=64, df_dim=64,
#         gfc_dim=1024, dfc_dim=1024, clr_dim=3, dataset_name='default',
#         input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None):
        #super ctor 
#        DCGAN.__init__(self, sess=sess, input_height=input_height, input_width=input_width, crop=crop, batch_size=batch_size, 
#                       sample_num=sample_num, output_height=output_height, output_width=output_width, y_dim=y_dim, z_dim=z_dim, 
#                       gf_dim=gf_dim, df_dim=df_dim, gfc_dim=gfc_dim, dfc_dim=dfc_dim, clr_dim=clr_dim, dataset_name=dataset_name, 
#                       input_fname_pattern=input_fname_pattern, checkpoint_dir=checkpoint_dir, sample_dir=sample_dir)
    def __init__(self, sess, 
                 flags, sample_num=64, y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, sample_dir=None):
        DCGAN.__init__(self, sess, flags, sample_num=sample_num, y_dim=y_dim, z_dim=z_dim, gf_dim=gf_dim, df_dim=df_dim,
                       gfc_dim=gfc_dim, dfc_dim=dfc_dim, sample_dir=sample_dir)
        
    def loadDataset(self):
        self.dbgOut('loadDataset')
        self.data = glob(os.path.join('./data', self.dataset_name, self.input_fname_pattern))
        imreadImg = imread(self.data[0]);
        if len(imreadImg.shape) >= 3: #check if image is a non-grayscale image by checking channel number
            self.clr_dim = imread(self.data[0]).shape[-1]
        else:
            self.clr_dim = 1


    def train(self, config):
        self.dbgOut('train')
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g_loss, var_list=self.g_vars)
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.g_sum = merge_summary([self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = merge_summary([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = SummaryWriter('./logs', self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))
    
        sample_files = self.data[0:self.sample_num]
        sample = [
                get_image(sample_file,
                    input_height=self.input_height,
                    input_width=self.input_width,
                    resize_height=self.output_height,
                    resize_width=self.output_width,
                    crop=self.crop,
                    grayscale=self.grayscale) for sample_file in sample_files]
        if (self.grayscale):
            sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            sample_inputs = np.array(sample).astype(np.float32)
  
        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            #checkpoint_counter is saved incorrectly
            counter = checkpoint_counter
            print(' [*] Load SUCCESS chkpt counter : %d '%(checkpoint_counter))
        else:
            print(' [!] Load failed...')

        for epoch in xrange(config.epoch):
            self.data = glob(os.path.join('./data', config.dataset, self.input_fname_pattern))
            batch_idxs = min(len(self.data), config.train_size) // config.batch_size
    
            for idx in xrange(0, batch_idxs):
                batch_files = self.data[idx*config.batch_size:(idx+1)*config.batch_size]
                batch = [
                        get_image(batch_file,
                            input_height=self.input_height,
                            input_width=self.input_width,
                            resize_height=self.output_height,
                            resize_width=self.output_width,
                            crop=self.crop,
                            grayscale=self.grayscale) for batch_file in batch_files]
                if self.grayscale:
                    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)
    
                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)
    
                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],feed_dict={ self.inputs: batch_images, self.z: batch_z })
                self.writer.add_summary(summary_str, counter)
    
                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum],feed_dict={ self.z: batch_z })
                self.writer.add_summary(summary_str, counter)
    
                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum],feed_dict={ self.z: batch_z })
                self.writer.add_summary(summary_str, counter)
              
                errD_fake = self.d_loss_fake.eval({ self.z: batch_z })
                errD_real = self.d_loss_real.eval({ self.inputs: batch_images })
                errG = self.g_loss.eval({self.z: batch_z})
    
                print('Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f' \
                      % (epoch, idx, batch_idxs,time.time() - start_time, errD_fake+errD_real, errG))
    
                if np.mod(counter, 100) == 0:
                    feedDict={self.z: sample_z, self.inputs: sample_inputs}
                    try:
                        self.sampleTrainImgs(config, epoch, idx, feedDict=feedDict)
                    except:
                        print('one pic error!...')
    
                if np.mod(counter, 500) == 0:
                    self.save(config.checkpoint_dir, counter)
    
                counter += 1

        
    def discriminator(self, image, y=None, reuse=False):
        self.dbgOut('discriminator')
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4), h4
 

    def generator(self, z, y=None):
        self.dbgOut('generator')
        with tf.variable_scope('generator') as scope:
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
    
            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)
    
            self.h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(self.h0))
    
            self.h1, self.h1_w, self.h1_b = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))
    
            h2, self.h2_w, self.h2_b = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))
    
            h3, self.h3_w, self.h3_b = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))
    
            h4, self.h4_w, self.h4_b = deconv2d(h3, [self.batch_size, s_h, s_w, self.clr_dim], name='g_h4', with_w=True)
    
            return tf.nn.tanh(h4)
 

    def sampler(self, z, y=None):
        self.dbgOut('sampler')
        with tf.variable_scope('generator') as scope:
            scope.reuse_variables()
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
    
            # project `z` and reshape
            h0 = tf.reshape(linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),[-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(h0, train=False))
    
            h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1, train=False))
    
            h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2, train=False))
    
            h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3, train=False))
    
            h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.clr_dim], name='g_h4')
    
            return tf.nn.tanh(h4)
        

#variant of DCGAN to handle conditional GAN mnist
class mnistDCGan(DCGAN):
    className = 'mnistDCGan'
    
#    def __init__(self, sess, input_height=108, input_width=108, crop=True,
#         batch_size=64, sample_num = 64, output_height=64, output_width=64,
#         y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
#         gfc_dim=1024, dfc_dim=1024, clr_dim=3, dataset_name='default',
#         input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None):
#        #super ctor 
#        DCGAN.__init__(self, sess=sess, input_height=input_height, input_width=input_width, crop=crop, batch_size=batch_size, 
#                       sample_num=sample_num, output_height=output_height, output_width=output_width, y_dim=y_dim, z_dim=z_dim, 
#                       gf_dim=gf_dim, df_dim=df_dim, gfc_dim=gfc_dim, dfc_dim=dfc_dim, clr_dim=clr_dim, dataset_name=dataset_name, 
#                       input_fname_pattern=input_fname_pattern, checkpoint_dir=checkpoint_dir, sample_dir=sample_dir)
    def __init__(self, sess, 
                 flags, sample_num=64, y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, sample_dir=None):
        DCGAN.__init__(self, sess, flags, sample_num=sample_num, y_dim=y_dim, z_dim=z_dim, gf_dim=gf_dim, df_dim=df_dim,
                       gfc_dim=gfc_dim, dfc_dim=dfc_dim, sample_dir=sample_dir)
        
    def loadDataset(self):
        self.dbgOut('loadDataset')
        self.data_X, self.data_y = self.load_mnist()
        self.clr_dim = self.data_X[0].shape[-1]
        
    def train(self, config):
        self.dbgOut('train')
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g_loss, var_list=self.g_vars)
        try:
          tf.global_variables_initializer().run()
        except:
          tf.initialize_all_variables().run()

        self.g_sum = merge_summary([self.z_sum, self.d__sum,self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = merge_summary([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = SummaryWriter('./logs', self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))
    
        sample_inputs = self.data_X[0:self.sample_num]
        sample_labels = self.data_y[0:self.sample_num]
  
        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            #checkpoint_counter is saved incorrectly
            counter = checkpoint_counter
            print(' [*] Load SUCCESS chkpt counter : %d '%(checkpoint_counter))
        else:
            print(' [!] Load failed...')

        for epoch in xrange(config.epoch):
            batch_idxs = min(len(self.data_X), config.train_size) // self.batch_size

            for idx in xrange(0, batch_idxs):
                batch_images = self.data_X[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_labels = self.data_y[idx*self.batch_size:(idx+1)*self.batch_size]

                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                
                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum], feed_dict={self.inputs: batch_images,self.z: batch_z,self.y:batch_labels})
                self.writer.add_summary(summary_str, counter)
              
                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum],feed_dict={self.z: batch_z, self.y:batch_labels})
                self.writer.add_summary(summary_str, counter)
              
                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum],feed_dict={ self.z: batch_z, self.y:batch_labels })
                self.writer.add_summary(summary_str, counter)
                
                errD_fake = self.d_loss_fake.eval({self.z: batch_z,self.y:batch_labels})
                errD_real = self.d_loss_real.eval({self.inputs: batch_images,self.y:batch_labels})
                errG = self.g_loss.eval({self.z: batch_z,self.y: batch_labels})

                print('Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f' \
                      % (epoch, idx, batch_idxs, time.time() - start_time, errD_fake+errD_real, errG))
            
                if np.mod(counter, 100) == 0:
                    try:
                        feedDict={self.z: sample_z, self.inputs: sample_inputs, self.y:sample_labels}
                        self.sampleTrainImgs(config, epoch, idx, feedDict=feedDict)
                    except:
                        print('one pic error!...')

                if np.mod(counter, 500) == 0:
                    self.save(config.checkpoint_dir, counter)

                counter += 1      
        
        
    def discriminator(self, image, y=None, reuse=False):
        #self.dbgOut('discriminator')
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            x = conv_cond_concat(image, yb)
            
            h0 = lrelu(conv2d(x, self.clr_dim + self.y_dim, name='d_h0_conv'))
            h0 = conv_cond_concat(h0, yb)
            
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
            h1 = tf.reshape(h1, [self.batch_size, -1])      
            h1 = concat([h1, y], 1)
            
            h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
            h2 = concat([h2, y], 1)
            
            h3 = linear(h2, 1, 'd_h3_lin')
            
            return tf.nn.sigmoid(h3), h3

    def generator(self, z, y):
        #self.dbgOut('generator')
        with tf.variable_scope('generator') as scope:
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_h4 = int(s_h/2), int(s_h/4)
            s_w2, s_w4 = int(s_w/2), int(s_w/4)
    
            # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            z = concat([z, y], 1)
    
            h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
            h0 = concat([h0, y], 1)
    
            h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin')))
            h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
    
            h1 = conv_cond_concat(h1, yb)
    
            h2 = tf.nn.relu(self.g_bn2(deconv2d(h1,[self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
            h2 = conv_cond_concat(h2, yb)
    
            return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.clr_dim], name='g_h3')) 
            
    def sampler(self, z, y):
        #self.dbgOut('sampler')
        with tf.variable_scope('generator') as scope:
            scope.reuse_variables()
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_h4 = int(s_h/2), int(s_h/4)
            s_w2, s_w4 = int(s_w/2), int(s_w/4)
            
            # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            #tf.concat call in ops.py
            z = concat([z, y], 1)
            
            h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=False))
            h0 = concat([h0, y], 1)
            
            h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin'), train=False))
            h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
            h1 = conv_cond_concat(h1, yb)
            
            h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
            h2 = conv_cond_concat(h2, yb)
            
            return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.clr_dim], name='g_h3'))
    
    def load_mnist(self):
        #self.dbgOut('load_mnist')
        data_dir = os.path.join('./data', self.dataset_name)
        
        fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)
        #Size of trx :  47040000  shape of trx : (60000, 28, 28, 1)
        
        fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.float)
        
        fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)
        
        fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.float)
        
        trY = np.asarray(trY)
        teY = np.asarray(teY)
        
        X = np.concatenate((trX, teX), axis=0)
        #Size of X :  54880000  shape of Xs : (70000, 28, 28, 1)
        y = np.concatenate((trY, teY), axis=0).astype(np.int)
        
        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)
        
        y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
        for i, label in enumerate(y):
            y_vec[i,y[i]] = 1.0
        
        return X/255.,y_vec

class gestCondDCGan(mnistDCGan):
    className = 'gestCondDCGan'
    
#    def __init__(self, sess, input_height=108, input_width=108, crop=True,
#         batch_size=64, sample_num = 64, output_height=64, output_width=64,
#         y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
#         gfc_dim=1024, dfc_dim=1024, clr_dim=3, dataset_name='default',
#         input_fname_pattern='*.png', checkpoint_dir=None, sample_dir=None):
#        
#        #build lists of alphabet directories and directories of csv files holding hand COM and COMVel locs
#        self.imgAlphabetLocs = baseDataDirList#[os.path.join(gestIKdataBaseDir, x) for x in baseDataDirList]
#        self.lblDataLocs = ['COMVals_'+x for x in baseDataDirList]#[os.path.join(gestIKdataBaseDir, ('COMVals_'+x)) for x in baseDataDirList]
#
#        #super ctor 
#        DCGAN.__init__(self, sess=sess, input_height=input_height, input_width=input_width, crop=crop, batch_size=batch_size, 
#                       sample_num=sample_num, output_height=output_height, output_width=output_width, y_dim=y_dim, z_dim=z_dim, 
#                       gf_dim=gf_dim, df_dim=df_dim, gfc_dim=gfc_dim, dfc_dim=dfc_dim, clr_dim=clr_dim, dataset_name=dataset_name, 
#                       input_fname_pattern=input_fname_pattern, checkpoint_dir=checkpoint_dir, sample_dir=sample_dir)

    def __init__(self, sess, 
                 flags, sample_num=64, y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, sample_dir=None):
        #build lists of alphabet directories and directories of csv files holding hand COM and COMVel locs before calling super ctor
        self.imgAlphabetLocs = baseDataDirList#[os.path.join(gestIKdataBaseDir, x) for x in baseDataDirList]
        self.lblDataLocs = ['COMVals_'+x for x in baseDataDirList]#[os.path.join(gestIKdataBaseDir, ('COMVals_'+x)) for x in baseDataDirList]

        DCGAN.__init__(self, sess, flags, sample_num=sample_num, y_dim=y_dim, z_dim=z_dim, gf_dim=gf_dim, df_dim=df_dim,
                       gfc_dim=gfc_dim, dfc_dim=dfc_dim, sample_dir=sample_dir)
                   

        
    def loadDataset(self):
        self.dbgOut('loadDataset')
        self.imgFNameList, self.yVecList = self.load_gestIK()
        print('done loading dataset : imgFNameList size : ' + str(len(self.imgFNameList)) + ' yVec size : ' + str(len(self.yVecList)))
        self.clr_dim = 1
        #imreadImg = imread(self.imgFNameList[0]);
#        if len(imreadImg.shape) >= 3: #check if image is a non-grayscale image by checking channel number
#            self.clr_dim = imread(self.imgList[0]).shape[-1]
#        else:
#            self.clr_dim = 1

#        for x in range(100):
#            print('F:%s | class : ' %(self.imgFNameList[x]), end='')
#            print(self.yVecList[x])

    def train(self, config):
        self.dbgOut('train')
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g_loss, var_list=self.g_vars)
        
        #try:
        tf.global_variables_initializer().run()
        #except:
        #  tf.initialize_all_variables().run()

        self.g_sum = merge_summary([self.z_sum, self.d__sum,self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = merge_summary([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = SummaryWriter('./logs', self.sess.graph)

        #samples for noise(z), conditional class/cntls (y) and comparative images (x)  used to generate saves
        sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))
        sample_files = self.imgFNameList[0:self.sample_num]
        sample = [
                get_image(sample_file,
                    input_height=self.input_height,
                    input_width=self.input_width,
                    resize_height=self.output_height,
                    resize_width=self.output_width,
                    crop=self.crop,
                    grayscale=self.grayscale) for sample_file in sample_files]
        if (self.grayscale):
            sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            sample_inputs = np.array(sample).astype(np.float32)
        
        sample_labels = self.yVecList[0:self.sample_num]        
        
        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            #checkpoint_counter is saved incorrectly
            counter = checkpoint_counter
            print(' [*] Loading pre-built model SUCCESS chkpt counter : %d '%(checkpoint_counter))
        else:
            print(' [!] Loading pre-built model failed, or no model found...')
            
        batch_idxs = min(len(self.imgFNameList), config.train_size) // self.batch_size

        for epoch in xrange(config.epoch):
            #batch_idxs = min(len(self.data_X), config.train_size) // self.batch_size
            #self.data = glob(os.path.join('./data', config.dataset, self.input_fname_pattern))
            #verify that this is necessary!
            #self.imgList, self.lblVals = self.load_gestIK()
            
            #batch_idxs = min(len(self.imgList), config.train_size) // self.batch_size

            for idx in xrange(0, batch_idxs):
                stRange = idx*self.batch_size
                endRange = (idx+1)*self.batch_size
                
                batch_files = self.imgFNameList[stRange:endRange]
                batch = [
                        get_image(batch_file,
                            input_height=self.input_height,
                            input_width=self.input_width,
                            resize_height=self.output_height,
                            resize_width=self.output_width,
                            crop=self.crop,
                            grayscale=self.grayscale) for batch_file in batch_files]
                if self.grayscale:
                    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)
                    
                batch_labels = self.yVecList[stRange:endRange]

                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                
                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum], feed_dict={self.inputs: batch_images,self.z: batch_z,self.y:batch_labels})
                self.writer.add_summary(summary_str, counter)
              
                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum],feed_dict={self.z: batch_z, self.y:batch_labels})
                self.writer.add_summary(summary_str, counter)
              
                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum],feed_dict={ self.z: batch_z, self.y:batch_labels })
                self.writer.add_summary(summary_str, counter)
                
                errD_fake = self.d_loss_fake.eval({self.z: batch_z,self.y:batch_labels})
                errD_real = self.d_loss_real.eval({self.inputs: batch_images,self.y:batch_labels})
                errG = self.g_loss.eval({self.z: batch_z,self.y: batch_labels})

                print('Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f' \
                      % (epoch, idx, batch_idxs, time.time() - start_time, errD_fake+errD_real, errG))
            
                if np.mod(counter, 10) == 0:
                    try:
                        feedDict={self.z: sample_z, self.inputs: sample_inputs, self.y:sample_labels}
                        self.sampleTrainImgs(config, epoch, idx, feedDict=feedDict)
                    except:
                        print('one pic error!...')

                if np.mod(counter, 500) == 0:
                    self.save(config.checkpoint_dir, counter)

                counter += 1   
                
                
    def build_model(self):
        self.dbgOut('build_model')
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

        #if self.crop:
        image_dims = [self.output_height, self.output_width, self.clr_dim]
        #else:
        #    image_dims = [self.input_height, self.input_width, self.clr_dim]

        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images')
        self.sample_inputs = tf.placeholder(tf.float32, [self.sample_num] + image_dims, name='sample_inputs')

        inputs = self.inputs
        sample_inputs = self.sample_inputs

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.z_sum = histogram_summary('z', self.z)

        self.G = self.generator(self.z, self.y)
        self.D, self.D_logits = self.discriminator(inputs, self.y, reuse=False)

        self.sampler = self.sampler(self.z, self.y)
        self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)

#        self.d_sum = histogram_summary('d', self.D)
#        self.d__sum = histogram_summary('d_', self.D_)
#        self.G_sum = image_summary('G', self.G)
        
        self.buildLossFuncs()

        #TODO: potentially conditionally build summary functionality, for speed concerns?
        self.buildSummaryFuncs()
    
#        self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
#        self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
#        self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
#    
#        self.d_loss_real_sum = scalar_summary('d_loss_real', self.d_loss_real)
#        self.d_loss_fake_sum = scalar_summary('d_loss_fake', self.d_loss_fake)
#                              
#        self.d_loss = self.d_loss_real + self.d_loss_fake
#    
#        self.g_loss_sum = scalar_summary('g_loss', self.g_loss)
#        self.d_loss_sum = scalar_summary('d_loss', self.d_loss)
    
        t_vars = tf.trainable_variables()
    
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
    
        self.saver = tf.train.Saver()
        

    def discriminator(self, image, y, reuse=False):
        self.dbgOut('discriminator ydim : ' + str(self.y_dim))
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            #shape of y : batch_size x 1 x 1 x 36
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            x = conv_cond_concat(image, yb)
            
            h0 = lrelu(conv2d(x, self.clr_dim + self.y_dim, name='d_h0_conv'))
            h0 = conv_cond_concat(h0, yb)
            
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
            h1 = tf.reshape(h1, [self.batch_size, -1])      
            h1 = concat([h1, y], 1)
            
            h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
            h2 = concat([h2, y], 1)
            
            h3 = linear(h2, 1, 'd_h3_lin')
            
            return tf.nn.sigmoid(h3), h3

    def generator(self, z, y):
        self.dbgOut('generator')
        with tf.variable_scope('generator') as scope:
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_h4 = int(s_h/2), int(s_h/4)
            s_w2, s_w4 = int(s_w/2), int(s_w/4)
    
            # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            z = concat([z, y], 1)
    
            h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
            h0 = concat([h0, y], 1)
    
            h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin')))
            h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
    
            h1 = conv_cond_concat(h1, yb)
    
            h2 = tf.nn.relu(self.g_bn2(deconv2d(h1,[self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
            h2 = conv_cond_concat(h2, yb)
    
            return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.clr_dim], name='g_h3')) 
            
    def sampler(self, z, y):
        self.dbgOut('sampler')
        with tf.variable_scope('generator') as scope:
            scope.reuse_variables()
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_h4 = int(s_h/2), int(s_h/4)
            s_w2, s_w4 = int(s_w/2), int(s_w/4)
            
            # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            #tf.concat call in ops.py
            z = concat([z, y], 1)
            
            h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=False))
            h0 = concat([h0, y], 1)
            
            h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin'), train=False))
            h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
            h1 = conv_cond_concat(h1, yb)
            
            h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
            h2 = conv_cond_concat(h2, yb)
            
            return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.clr_dim], name='g_h3'))


        
    def getFrameNum(self, imgIDX, numImgs):
        lastImgIDX = numImgs-1
        iVal = 9.0 * (imgIDX/(1.0*lastImgIDX))
        interpVal, stIdxf = math.modf(iVal)
        #mapping to idxs 0-10
        stIdx = int(stIdxf)
        return stIdx, (1.0-interpVal),stIdx+1, interpVal
        
    #return class vector with first 26 spots being 1hot for class, and last 10 spots being binary encoded frame
    def buildYVec(self, clsIDX, stIdx, stInterp, endIdx, endInterp):
        imgY_vec = np.zeros(36) #TODO replace with self.y_dim
        imgY_vec[clsIDX] = 1            #set letter value as class
        if stInterp > 0 :
            imgY_vec[26 + stIdx] = stInterp
        if endInterp > 0 :
            imgY_vec[26 + endIdx] = endInterp
        return imgY_vec        
        

    def load_gestIK(self):
        self.dbgOut('load_gestIK')
        #remake file
        #self.buildListFromDirListing(os.path.join('.','gestIK'),self.imgAlphabetLocs)
        imgFNameList = []#list of img file name strings
        y_vec = []#list of np arrays length 36 holding class value as 1-hot and 10-value interpolated frame location vector
        #format of entry in filename gestIKdataFileName : <fully qualified filename>|<letter as int>|<st idx>|<st interp val>|<end idx>|<end interp val>
        #ie : E:\Dropbox\Public\GestureIK\frames\CVEL_06081106\06081106_w_00414_CVEL\06081106_w_00414_CVEL.0000.png|22|0|1.0000|1|0.0000  
        src_lines = readFileDat(gestIKdataFileName)
        for line in src_lines:
            elemList = line.split('|')
            imgFNameList.append(elemList[0])
            y_vec.append(self.buildYVec(int(elemList[1]),int(elemList[2]),float(elemList[3]),int(elemList[4]),float(elemList[5])))
        return imgFNameList,y_vec
        
    
    #build file holding all letter directory names from all subdir elements in baseSubDirs list
    #baseSubDirs is list of all alphabet directories
    #should only run 1 time when using new data - writes files that can be used instead
    def buildListFromDirListing(self, dataDir, baseSubDirs):
        #find location of class encoded in example name - regex ptrn
        pattern = re.compile('_[a-z]_')
        X_YimgFNames = []
        chOffset = ord('a')
        for alphaDir in baseSubDirs:
            print('alphaDir : ',alphaDir)
            #each alphaDir expected to hold results from single data gen run
            ltrSubDirs = glob(os.path.join(gestIKdataBaseDir, alphaDir, '*'))
            for ltrPngSubDir in ltrSubDirs:
                if '.txt' in ltrPngSubDir :
                    continue   
                #get all image file names in list
                imgList = glob(ltrPngSubDir + os.sep + '*.png')
                #imgList = next(os.walk(ltrPngSubDir))[2]
                #list of images for a particular letter
                dList = os.path.normpath(ltrPngSubDir).split(os.sep)
                #class is in dList[-1] - will be element in list with only 1 member
                srch = re.search(pattern, dList[-1])
                cls = srch.group(0)[1]
                #build class vector for each element by concatenating 26 1-hot class elements and position in sequence
                intCls = ord(cls)-chOffset
                strCls = str(intCls)
                numImgs = len(imgList)
    
                for img in imgList: 
                    #get frame number of image
                    imgNum = int(img.split('.')[-2])
                    stIdx, stInterp, endIdx, endInterp = self.getFrameNum(imgNum,numImgs)
                    exStr = img + '|' + strCls + '|' + ('%d|%.4f|%d|%.4f' % (stIdx, stInterp, endIdx, endInterp))+ '\n'
                    X_YimgFNames.append(exStr)
        
        shuffle(X_YimgFNames)
        writeFileDat(os.path.join(dataDir,'allImgFileNamesAndClass.txt'), X_YimgFNames)
        print('buildListFromDirListing : done building imgnamelist')  
    
    
    
    
    
    
    
    
    