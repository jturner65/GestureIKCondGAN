from __future__ import division
import os
import time
import re
#import math
from glob import glob
import tensorflow as tf
import numpy as np
#xrange slightly faster than range - returns a generator of values instead of a list
from six.moves import xrange

from abc import ABC, abstractmethod

from ops import *
from utils import *


#def conv_out_size_same(size, stride):
#    return int(math.ceil(float(size) / float(stride)))
#
#def sigmoid_cross_entropy_with_logits(x, y):
#    try:
#        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
#    except:
#        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)


class baseDCGAN(ABC,object):
    className = 'baseDCGAN (abstract)'
    def __init__(self, sess, flags, 
                 sample_num=64, y_dim=None, z_dim=100, 
                 gf_dim=64, df_dim=64, gfc_dim=1024, dfc_dim=1024, 
                 sample_dir=None):
        """
        Args:
          sess: TensorFlow session
          batch_size: The size of batch. Should be specified before training.
          sample_num : (optional) Number of samples to take during training
          y_dim: (optional) Size of dim for y (conditional GAN). [None]
          z_dim: (optional) Size of dim for Z. [100]
          c_dim: (optional) Dimension(s) of 
          gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
          df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
          gfc_dim: (optional) Dimension of gen units for fully connected layer. [1024]
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
        
        self.train_size = flags.train_size
        
        #resize input images
        self.resize = flags.resize
        if self.resize : 
            self.resize_in_h = flags.resize_height
            self.resize_in_w = flags.resize_width
        else :
            self.resize_in_h = flags.output_height
            self.resize_in_w = flags.output_width
        
        self.dataset_name = flags.dataset
        self.input_fname_pattern = flags.input_fname_pattern
        self.crop = flags.crop        
        self.checkpoint_dir = flags.checkpoint_dir
        
        self.y_dim = y_dim
        self.z_dim = z_dim
        
        #gf_dim/df_dim: Dimension of gen/disc filters in first conv layer. [defaults to 64]
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        #gfc_dim/dfc_dim: Dimension of gen/disc units for fully connected layer. [defaults to 1024]
        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim
        
        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        #don't use for mnist
        if self.dataset_name != 'mnist':
        #if not self.y_dim:
            self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        
        if self.dataset_name != 'mnist':
        #if not self.y_dim:
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
        elif self.resize:
            image_dims = [self.resize_in_h, self.resize_in_w, self.clr_dim]
        else:
            image_dims = [self.input_height, self.input_width, self.clr_dim]
        
        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images')
        #self.sample_inputs = tf.placeholder(tf.float32, [self.sample_num] + image_dims, name='sample_inputs')
        
        inputs = self.inputs
        #sample_inputs = self.sample_inputs
        
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        #self.z_sum = tf.summary.histogram('z', self.z)
        #if condGAN
        if self.y_dim:
            self.G = self.generator(self.z, self.y)
            self.D, self.D_logits = self.discriminator(inputs, self.y, reuse=False)
        
            self.sampler = self.sampler(self.z, self.y)
            self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)
            
        else:
            self.G = self.generator(self.z)
            self.D, self.D_logits = self.discriminator(inputs)
        
            self.sampler = self.sampler(self.z)
            self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)        
          
        self.buildLossFuncs()
        
        #TODO: potentially build summary functionality only conditionally, for speed concerns?
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
    
#        self.d_loss_real_sum = tf.summary.scalar('d_loss_real', self.d_loss_real)
#        self.d_loss_fake_sum = tf.summary.scalar('d_loss_fake', self.d_loss_fake)
                              
        self.d_loss = self.d_loss_real + self.d_loss_fake
  
    def buildSummaryFuncs(self):
        #summary variables used for logging - can build conditionally if speed is an issue
        self.z_sum = tf.summary.histogram('z', self.z)

        self.d_loss_real_sum = tf.summary.scalar('d_loss_real', self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar('d_loss_fake', self.d_loss_fake)
        
        self.d_sum = tf.summary.histogram('d', self.D)
        self.d__sum = tf.summary.histogram('d_', self.D_)
        self.G_sum = tf.summary.image('G', self.G)
        self.g_loss_sum = tf.summary.scalar('g_loss', self.g_loss)
        self.d_loss_sum = tf.summary.scalar('d_loss', self.d_loss)

    
    def buildTrainSummary(self):
        self.g_sum = tf.summary.merge([self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.summary.FileWriter('./logs', self.sess.graph)

    @abstractmethod
    def train(self, config):
        pass
    
    #save a sample image of the current state of training
    def sampleTrainImgs(self, config, epoch, idx, feedDict):
        try:  
            samples, d_loss, g_loss = self.sess.run([self.sampler, self.d_loss, self.g_loss], feed_dict=feedDict)
            manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
            manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
            if (self.sample_dir != None):
                save_images(samples, [manifold_h, manifold_w], './{}/train_{:02d}_{:04d}.png'.format(self.sample_dir, epoch, idx))
            else :
                save_images(samples, [manifold_h, manifold_w], './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
            print('[Sample] d_loss: %.8f, g_loss: %.8f' % (d_loss, g_loss)) 
        except:
            print('Error saving picture in {} : epoch : {:02d} batch idx : {:04d}'.format(self.sample_dir, epoch, idx))
    
    #get a batch of images
    def getImageBatch(self, batch_files, resize_h, resize_w):
        batch = [
                get_image(batch_file,
                    input_height=self.input_height, input_width=self.input_width,
                    resize_height=resize_h, resize_width=resize_w,
                    crop=self.crop, grayscale=self.grayscale) 
                        for batch_file in batch_files]
        if self.grayscale:
            batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
        else:
            batch_images = np.array(batch).astype(np.float32)
        return batch_images

      
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
        
    #save current trained model at step
    def saveModel(self, step):
        model_name = self.className + '.model'
        print('save model name :' + model_name + ' step : ' + str(step))
        save_Chkpt_Dir = os.path.join(self.checkpoint_dir, self.model_dir)

        if not os.path.exists(save_Chkpt_Dir):
            os.makedirs(save_Chkpt_Dir)

        self.saver.save(self.sess,
            os.path.join(save_Chkpt_Dir, model_name),
            global_step=step)
                
    #check if model exists, load if so 
    #set starting epoch and batch idx based on saved model
    def checkForModelAndLoad(self):
        counter = 0
        epochStart = 0
        fstBatchStart = 0        
        print(' [*] Reading checkpoints...')
        chkptDir = os.path.join(self.checkpoint_dir, self.model_dir)
        ckpt = tf.train.get_checkpoint_state(chkptDir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            #print('ckpt_name :' + ckpt_name)
            self.saver.restore(self.sess, os.path.join(chkptDir, ckpt_name))
            #checkpoint_counter is iteration value of saved model - need to start on next iteration 
            counter = 1 + int(next(re.finditer('(\d+)(?!.*\d)',ckpt_name)).group(0))
            #print(' [*] Success reading {}'.format(ckpt_name))
            #epochStart : epoch to start retraining saved model
            epochStart =  counter // self.batch_idxs
            #fstBatchStart : first batch to start on for this first epoch, so as to not retrain on any batches (1 past saved batch)
            fstBatchStart = counter % self.batch_idxs 
            print(' [*] Loading pre-built model SUCCESS model name : %s starting counter : %d  epoch %d batch %d '%(ckpt_name, counter, epochStart, fstBatchStart))          
            return True, counter, epochStart, fstBatchStart
        else:
            print(' [!] Loading pre-built model failed, or no model found...')
            return False, counter, epochStart, fstBatchStart


#basic DCGAN implementation
class DCGan(baseDCGAN):
    className = 'DCGan'

    def __init__(self, sess, 
                 flags, sample_num=64, y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, sample_dir=None):
        
        baseDCGAN.__init__(self, sess, flags, sample_num=sample_num, y_dim=y_dim, z_dim=z_dim, gf_dim=gf_dim, df_dim=df_dim,
                       gfc_dim=gfc_dim, dfc_dim=dfc_dim, sample_dir=sample_dir)
        
        
    def loadDataset(self):
        self.dbgOut('loadDataset')
        tmpPath = os.path.join('./data', self.dataset_name, self.input_fname_pattern)
        #print('tmpPath to read : {}'.format(tmpPath))
        self.data = glob(tmpPath)

        imreadImg = imread(self.data[0]);
        if len(imreadImg.shape) >= 3: #check if image is a non-grayscale image by checking channel number
            self.clr_dim = imread(self.data[0]).shape[-1]
        else:
            self.clr_dim = 1
        #holds # of batch idxs (# of batches)
        self.batch_idxs = min(len(self.data), self.train_size) // self.batch_size


    def train(self, config):
        self.dbgOut('train')
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g_loss, var_list=self.g_vars)
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()
            
        self.buildTrainSummary()

        sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))
    
        sample_files = self.data[0:self.sample_num]
        sample_inputs = self.getImageBatch(sample_files, self.output_height, self.output_width)     
  
        start_time = time.time()        
        #batch_idxs = min(len(self.data), self.train_size) // self.batch_size
        #exists, counter, epochStart, fstBatchStart = self.checkForModelAndLoad(self.batch_idxs)
        exists, counter, epochStart, fstBatchStart = self.checkForModelAndLoad()
            
        self.data = glob(os.path.join('./data', config.dataset, self.input_fname_pattern))
        #save pics 4 times per epoch
        pic_save_loc = 1 + self.batch_idxs // 10
        #save model every 2 images
        mdl_save_loc = pic_save_loc * 2
        for epoch in xrange(epochStart, config.epoch):
            print("Save picture every %d batches, save model every %d batches" %(pic_save_loc,mdl_save_loc))
            for idx in xrange(fstBatchStart, self.batch_idxs):
                batch_files = self.data[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_images = self.getImageBatch(batch_files, self.output_height, self.output_width)
    
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
    
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
                      % (epoch, idx, self.batch_idxs,time.time() - start_time, errD_fake+errD_real, errG))
    
                if np.mod(idx, pic_save_loc) == 0:
                    feedDict={self.z: sample_z, self.inputs: sample_inputs}
                    self.sampleTrainImgs(config, epoch, idx, feedDict=feedDict)
    
                if np.mod(idx, mdl_save_loc) == 0:
                    self.saveModel(counter)
    
                counter += 1
            #reset to start at first batch each subsequent epoch
            fstBatchStart = 0
        #save last model
        self.saveModel(counter)

        
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
class mnistDCGan(baseDCGAN):
    className = 'mnistDCGan'
    
    def __init__(self, sess, 
                 flags, sample_num=64, y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, sample_dir=None):
        baseDCGAN.__init__(self, sess, flags, sample_num=sample_num, y_dim=y_dim, z_dim=z_dim, gf_dim=gf_dim, df_dim=df_dim,
                       gfc_dim=gfc_dim, dfc_dim=dfc_dim, sample_dir=sample_dir)
        
    def loadDataset(self):
        self.dbgOut('loadDataset')
        self.data_X, self.data_y = self.load_mnist()
        self.clr_dim = self.data_X[0].shape[-1]
        #holds # of batch idxs (# of batches)
        self.batch_idxs = min(len(self.data_X), self.train_size) // self.batch_size
        
    def train(self, config):
        self.dbgOut('train')
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g_loss, var_list=self.g_vars)
        try:
          tf.global_variables_initializer().run()
        except:
          tf.initialize_all_variables().run()

        self.buildTrainSummary()
        
        #set up sample for each save picture - uses same data so that can see how model evolves
        sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))    
        sample_inputs = self.data_X[0:self.sample_num]
        sample_labels = self.data_y[0:self.sample_num]
  
        start_time = time.time()
        #batch_idxs = min(len(self.data_X), self.train_size) // self.batch_size
        #exists, counter, epochStart, fstBatchStart = self.checkForModelAndLoad(self.batch_idxs)
        exists, counter, epochStart, fstBatchStart = self.checkForModelAndLoad()

        #save pics 4 times per epoch
        pic_save_loc = 1 + self.batch_idxs // 4
        #save model every 2 images
        mdl_save_loc = pic_save_loc * 2
        for epoch in xrange(epochStart, config.epoch):
            print("Save picture every %d batches, save model every %d batches" %(pic_save_loc,mdl_save_loc))
            for idx in xrange(fstBatchStart, self.batch_idxs):
                stIdx = idx*self.batch_size
                endIdx = (idx+1)*self.batch_size
                batch_images = self.data_X[stIdx:endIdx]
                batch_labels = self.data_y[stIdx:endIdx]

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
                      % (epoch, idx, self.batch_idxs, time.time() - start_time, errD_fake+errD_real, errG))
            
                if np.mod(idx, pic_save_loc) == 0:
                    feedDict={self.z: sample_z, self.inputs: sample_inputs, self.y:sample_labels}
                    self.sampleTrainImgs(config, epoch, idx, feedDict=feedDict)

                if np.mod(idx, mdl_save_loc) == 0:
                    self.saveModel(counter)

                counter += 1      
            #reset to start at first batch each subsequent epoch
            fstBatchStart = 0
        #save last model
        self.saveModel(counter)
        
        
    def discriminator(self, image, y=None, reuse=False):
        #self.dbgOut('discriminator')
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            x = conv_cond_concat(image, yb)
            
            h0 = lrelu(conv2d(x, self.clr_dim + self.y_dim, name='d_h0_conv'))
            #h0 = conv_cond_concat(h0, yb)
            
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
            h1 = tf.reshape(h1, [self.batch_size, -1])      
            #h1 = concat([h1, y], 1)
            
            h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
            #h2 = concat([h2, y], 1)
                  
            
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
            #h0 = concat([h0, y], 1)
    
            h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin')))
            h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])    
            #h1 = conv_cond_concat(h1, yb)
    
            h2 = tf.nn.relu(self.g_bn2(deconv2d(h1,[self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
            #h2 = conv_cond_concat(h2, yb)
    
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
            #h0 = concat([h0, y], 1)
            
            h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin'), train=False))
            h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
            #h1 = conv_cond_concat(h1, yb)
            
            h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
            #h2 = conv_cond_concat(h2, yb)
            
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

