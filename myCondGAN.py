from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
#import re
#xrange slightly faster than range - returns a generator of values instead of a list
from six.moves import xrange

from random import shuffle

from myBaseModels import baseDCGAN

from ops import *
from utils import *


#gestIK data variables - base directory and sub dirs holding training data
gestIKdataBaseDir = 'F:\\Dropbox\\Public\\GestureIK\\frames'
baseDataDirList = ['CVEL_06261746']#['CVEL_06101316','CVEL_06081106']

#whether or not to regenerate base data file - set true only upon initial run after new data generated
#base data file is listing of all individual examples in all sequeneces, randomized
regenBaseDataFile = False

#class for conditional DCGAN 
class gestCondDCGan(baseDCGAN):
    className = 'gestCondDCGan'

    def __init__(self, sess, 
                 flags, sample_num=64, y_dim=None, z_dim=100, 
                 gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, 
                 sample_dir=None):
        
        #build lists of alphabet directories and directories of csv files holding hand COM and COMVel locs before calling super ctor
        self.imgAlphabetLocs = baseDataDirList#[os.path.join(gestIKdataBaseDir, x) for x in baseDataDirList]
        self.lblDataLocs = ['COMVals_'+x for x in baseDataDirList]#[os.path.join(gestIKdataBaseDir, ('COMVals_'+x)) for x in baseDataDirList]
        
        self.imgListFileName = self.buildImgListFileName(y_dim)
        
        baseDCGAN.__init__(self, sess, flags, sample_num=sample_num, y_dim=y_dim, z_dim=z_dim, gf_dim=gf_dim, df_dim=df_dim,
                       gfc_dim=gfc_dim, dfc_dim=dfc_dim, sample_dir=sample_dir)
                   

    #loads list of file names, along with list of conditional flags for each file name
    def loadDataset(self):
        self.dbgOut('loadDataset')
        if regenBaseDataFile :
            #set regenBaseDataFile if new train data is available            
            print('Building file holding per-frame file names and key values' )
            buildListFromDirListing(os.path.join('.','gestIK'),self.imgAlphabetLocs, gestIKdataBaseDir, self.imgListFileName)
        self.imgFNameList, self.yVecList = self.load_gestIK()
        print('done loading dataset : imgFNameList size : ' + str(len(self.imgFNameList)) + ' yVec size : ' + str(len(self.yVecList)))
        self.batch_idxs = min(len(self.yVecList), self.train_size) // self.batch_size        
        self.clr_dim = 1
        self.grayscale = True

    #build the file name (without path) for the image list being used by this GAN
    #holds every image's fully qualified file name along with y-vector class info
    def buildImgListFileName(self, y_dim):
        if y_dim == None:
            y_dim = 0
        numAlphas = 0
        #build the file name by concatenating # of alphabets, base filename of 1st alphabet in list, and size of y_dim
        for alphaDir in self.imgAlphabetLocs:
            #each alphaDir expected to hold results from single data gen run
            ltrSubDirs = glob(os.path.join(gestIKdataBaseDir, alphaDir, '*'))
            numAlphas += len(ltrSubDirs)//26
        
        res = 'y_'+str(y_dim)+'_nABC_' + str(numAlphas)+'_'+self.imgAlphabetLocs[0].split('_')[-1]+'_imgFNames.txt'
        print (res)
        return res
        

    def train(self, config):
        self.dbgOut('train')
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g_loss, var_list=self.g_vars)
        
        #try:
        tf.global_variables_initializer().run()
        #except:
        #  tf.initialize_all_variables().run()

        self.buildTrainSummary()

        #samples for noise(z), conditional class/cntls (y) and comparative images (x)  used to generate saves
        sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))
        #get first sample_num examples of images and labels to use as save pic sample
        sample_inputs = self.getImageBatch(self.imgFNameList[0:self.sample_num], self.resize_in_h, self.resize_in_w)
        sample_labels = self.yVecList[0:self.sample_num] 
        
        start_time = time.time()        
        #batch_idxs = min(len(self.yVecList), self.train_size) // self.batch_size        
        #exists, counter, epochStart, fstBatchStart = self.checkForModelAndLoad(self.batch_idxs)
        exists, counter, epochStart, fstBatchStart = self.checkForModelAndLoad()

        #save pics 20 times per epoch
        pic_save_loc = 20 #1 + self.batch_idxs // 40
        #save model every 4 images
        mdl_save_loc = pic_save_loc * 4        
        
        for epoch in xrange(epochStart,config.epoch):
            print('Save picture every %d batches, save model every %d batches' %(pic_save_loc,mdl_save_loc))
            for idx in xrange(fstBatchStart, self.batch_idxs):
                stRange = idx*self.batch_size
                endRange = (idx+1)*self.batch_size
                
                batch_images = self.getImageBatch(self.imgFNameList[stRange:endRange], self.resize_in_h, self.resize_in_w)                   
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
                      % (epoch, idx, self.batch_idxs, time.time() - start_time, errD_fake+errD_real, errG))
            
                #if np.mod(counter, 100) == 0:
                if np.mod(idx, pic_save_loc) == 0:
                    feedDict={self.z: sample_z, self.inputs: sample_inputs, self.y:sample_labels}
                    self.sampleTrainImgs(config, epoch, idx, feedDict=feedDict)

                #if np.mod(counter, 500) == 0:
                if np.mod(idx, mdl_save_loc) == 0:
                    self.saveModel(counter)

                counter += 1 
            #next epoch starts with batch idx == 0
            fstBatchStart = 0
        #save final model
        self.saveModel(counter)                
                
    def build_model(self):
        self.dbgOut('build_model')
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')


        #input image dims
        if self.crop:
            #center crop of images
            image_dims = [self.output_height, self.output_width, self.clr_dim]
        elif self.resize:
            image_dims = [self.resize_in_h, self.resize_in_w, self.clr_dim]
        else:
            image_dims = [self.input_height, self.input_width, self.clr_dim]
            

        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images')
        inputs = self.inputs

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        self.G = self.generator(self.z, self.y)
        self.sampler = self.sampler(self.z, self.y)

        self.D, self.D_logits = self.discriminator(inputs, self.y, reuse=False)
        self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)
        
        #build loss functions 
        self.buildLossFuncs()

        #TODO: potentially conditionally build summary functionality, for speed concerns?
        self.buildSummaryFuncs()
    
        t_vars = tf.trainable_variables()
    
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
    
        self.saver = tf.train.Saver()

    def discriminator(self, image, y, reuse=False):
        self.dbgOut('discriminator')
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            #shape of y : batch_size x 1 x 1 x 36
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            x = conv_cond_concat(image, yb)

            h0 = lrelu(conv2d(x, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
            #h4 is logits
            return tf.nn.sigmoid(h4), h4
 

    def generator(self, z, y):
        self.dbgOut('generator')
        with tf.variable_scope('generator') as scope:
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
            
            z = concat([z, y], 1)
    
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
    
#            return tf.nn.sigmoid(h4) #alternate?
            return tf.nn.tanh(h4)
 

    def sampler(self, z, y):
        self.dbgOut('sampler')
        with tf.variable_scope('generator') as scope:
            scope.reuse_variables()
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
    
            z = concat([z, y], 1)
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
    
#            return tf.nn.sigmoid(h4) #alternate?
            return tf.nn.tanh(h4)        

        
    #return class vector with first 26 spots being 1hot for class, and last 10 spots being binary encoded frame
    def buildYVec(self, clsIDX, stIdx, stInterp, endIdx, endInterp):
        imgY_vec = np.zeros(self.y_dim) 
        imgY_vec[clsIDX] = 1            #set letter value as class
        if stInterp > 0 :
            imgY_vec[26 + stIdx] = stInterp
        if endInterp > 0 :
            imgY_vec[26 + endIdx] = endInterp
        return imgY_vec        
        
    def buildCamYVec(self, elemList):
        imgY_vec = self.buildYVec(int(elemList[1]),int(elemList[2]),float(elemList[3]),int(elemList[4]),float(elemList[5]))
        camTrans = elemList[6].split(',')
        for i in range(len(camTrans)):
            imgY_vec[36+i] = float(camTrans[i])
        imgY_vec[39] = float(elemList[7])
        camOrient = elemList[8].split(',')
        for i in range(len(camOrient)):
            imgY_vec[40+i] = float(camOrient[i])
        
        return imgY_vec        
        

    def load_gestIK(self):
        self.dbgOut('load_gestIK')
        #remake file
        #self.buildListFromDirListing(os.path.join('.','gestIK'),self.imgAlphabetLocs)
        imgFNameList = []#list of img file name strings
        yVecList = []#list of np arrays length 36 holding class value as 1-hot and 10-value interpolated frame location vector        
        gestIKdataFullPath = os.path.join('.','gestIK', self.imgListFileName)        
        #format of entry in filename gestIKdataFullPath : 
        #<fully qualified filename>|<letter as int>|<st idx>|<st interp val>|<end idx>|<end interp val>|<string of environment varables sep by pipes>
        src_lines = readFileDat(gestIKdataFullPath)
        if self.dataset_name == 'gestIKCond':  #simple dataset with only 36 key values  
            for line in src_lines:
                elemList = line.split('|')
                imgFNameList.append(elemList[0])
                yVecList.append(self.buildYVec(int(elemList[1]),int(elemList[2]),float(elemList[3]),int(elemList[4]),float(elemList[5])))
        elif self.dataset_name == 'gestIKCond2' :
            for line in src_lines:
                elemList = line.split('|')
                imgFNameList.append(elemList[0])
                #cam vals in idxs 6,7,8
                yVecList.append(self.buildCamYVec(elemList))
        else :
            print ('load_gestIK : Dataset %s not found' %(self.dataset_name))

        return imgFNameList,yVecList
        


#class infoCondDCGAN(gestCondDCGan):
#    className = 'infoCondDCGAN'
#
#    def __init__(self, sess, 
#                 flags, sample_num=64, 
#                 y_dim=None, z_dim=100, c_dim = None,
#                 gf_dim=64, df_dim=64,
#                 gfc_dim=1024, dfc_dim=1024, 
#                 sample_dir=None):
#        
#        self.c_dim = c_dim
#        gestCondDCGan.__init__(self, sess, flags, sample_num=sample_num, y_dim=y_dim, z_dim=z_dim, gf_dim=gf_dim, df_dim=df_dim,
#                       gfc_dim=gfc_dim, dfc_dim=dfc_dim, sample_dir=sample_dir)
#    
#
#    def build_model(self):
#        self.dbgOut('build_model')
#        self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
#        self.yc = tf.placeholder(tf.float32, [self.batch_size, self.y_dim + self.c_dim], name='yc')
#        #self.c = tf.placeholder(tf.float32, [self.batch_size, self.c_dim], name='c')
#
#
#        #input image dims
#        if self.crop:
#            #center crop of images
#            image_dims = [self.output_height, self.output_width, self.clr_dim]
#        elif self.resize:
#            image_dims = [self.resize_in_h, self.resize_in_w, self.clr_dim]
#        else:
#            image_dims = [self.input_height, self.input_width, self.clr_dim]
#            
#
#        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images')
#        
#        inputs = self.inputs
#
#        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
#
#        self.G = self.generator(self.z, self.yc)
#        self.sampler = self.sampler(self.z, self.yc)
#
#        self.D, self.D_logits = self.discriminator(inputs, self.y, reuse=False)
#        self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)
#        
#        self.Q_, self.Q_logits_ = self.discrQ(self.G, self.y)
#        
#        #build loss functions 
#        self.buildLossFuncs()
#
#        #TODO: potentially conditionally build summary functionality, for speed concerns?
#        self.buildSummaryFuncs()
#    
#        t_vars = tf.trainable_variables()
#    
#        self.d_vars = [var for var in t_vars if 'd_' in var.name]
#        self.g_vars = [var for var in t_vars if 'g_' in var.name]
#    
#        self.saver = tf.train.Saver()
#        
#    #override base function to include Q reference    
#    #build loss functions - override if desire other loss format
#    def buildLossFuncs(self):
#        self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
#        self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
#        self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
#    
##        #entropy definition : - sum of p(c) * log p(c)
##        condEnt = tf.reduce_mean(-tf.reduce_sum(tf.log(Qc_x + eps) * y, axis=1))
##        priorEnt = tf.reduce_mean(-tf.reduce_sum(tf.log(y + eps) * y, axis=1))
##        #investigate loss
##        Q_loss = priorEnt + condEnt 
#
##        self.d_loss_real_sum = tf.summary.scalar('d_loss_real', self.d_loss_real)
##        self.d_loss_fake_sum = tf.summary.scalar('d_loss_fake', self.d_loss_fake)
#                              
#        self.d_loss = self.d_loss_real + self.d_loss_fake
#  
#    #override base function to include Q reference    
#    def buildSummaryFuncs(self):
#        #summary variables used for logging - can build conditionally if speed is an issue
#        self.z_sum = tf.summary.histogram('z', self.z)
#
#        self.d_loss_real_sum = tf.summary.scalar('d_loss_real', self.d_loss_real)
#        self.d_loss_fake_sum = tf.summary.scalar('d_loss_fake', self.d_loss_fake)
#        
#        self.d_sum = tf.summary.histogram('d', self.D)
#        self.d__sum = tf.summary.histogram('d_', self.D_)
#        self.G_sum = tf.summary.image('G', self.G)
#        self.g_loss_sum = tf.summary.scalar('g_loss', self.g_loss)
#        self.d_loss_sum = tf.summary.scalar('d_loss', self.d_loss)
#        
#    
#    #override condDCGan training
#    def train(self, config):
#        self.dbgOut('train')
#        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.d_loss, var_list=self.d_vars)
#        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g_loss, var_list=self.g_vars)
#        
#        #try:
#        tf.global_variables_initializer().run()
#        #except:
#        #  tf.initialize_all_variables().run()
#
#        self.buildTrainSummary()
#
#        #samples for noise(z), conditional class/cntls (y) and comparative images (x)  used to generate saves
#        sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))
#        #get first sample_num examples of images and labels to use as save pic sample
#        sample_inputs = self.getImageBatch(self.imgFNameList[0:self.sample_num], self.resize_in_h, self.resize_in_w)
#        sample_labels = self.yVecList[0:self.sample_num] 
#        
#        start_time = time.time()        
#        batch_idxs = min(len(self.yVecList), self.train_size) // self.batch_size        
#        exists, counter, epochStart, fstBatchStart = self.checkForModelAndLoad(batch_idxs)
#
#        #save pics 20 times per epoch
#        pic_save_loc = 20 #1 + batch_idxs // 40
#        #save model every 4 images
#        mdl_save_loc = pic_save_loc * 4        
#        
#        for epoch in xrange(epochStart,config.epoch):
#            print('Save picture every %d batches, save model every %d batches' %(pic_save_loc,mdl_save_loc))
#            for idx in xrange(fstBatchStart, batch_idxs):
#                stRange = idx*self.batch_size
#                endRange = (idx+1)*self.batch_size
#                
#                batch_images = self.getImageBatch(self.imgFNameList[stRange:endRange], self.resize_in_h, self.resize_in_w)                   
#                batch_labels = self.yVecList[stRange:endRange]
#
#                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
#                
#                # Update D network
#                _, summary_str = self.sess.run([d_optim, self.d_sum], feed_dict={self.inputs: batch_images,self.z: batch_z,self.y:batch_labels})
#                self.writer.add_summary(summary_str, counter)
#              
#                # Update G network
#                _, summary_str = self.sess.run([g_optim, self.g_sum],feed_dict={self.z: batch_z, self.y:batch_labels})
#                self.writer.add_summary(summary_str, counter)
#              
#                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
#                _, summary_str = self.sess.run([g_optim, self.g_sum],feed_dict={ self.z: batch_z, self.y:batch_labels })
#                self.writer.add_summary(summary_str, counter)
#                
#                errD_fake = self.d_loss_fake.eval({self.z: batch_z,self.y:batch_labels})
#                errD_real = self.d_loss_real.eval({self.inputs: batch_images,self.y:batch_labels})
#                errG = self.g_loss.eval({self.z: batch_z,self.y: batch_labels})
#
#                print('Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f' \
#                      % (epoch, idx, batch_idxs, time.time() - start_time, errD_fake+errD_real, errG))
#            
#                #if np.mod(counter, 100) == 0:
#                if np.mod(idx, pic_save_loc) == 0:
#                    feedDict={self.z: sample_z, self.inputs: sample_inputs, self.y:sample_labels}
#                    self.sampleTrainImgs(config, epoch, idx, feedDict=feedDict)
#
#                #if np.mod(counter, 500) == 0:
#                if np.mod(idx, mdl_save_loc) == 0:
#                    self.saveModel(counter)
#
#                counter += 1 
#            #next epoch starts with batch idx == 0
#            fstBatchStart = 0
#        #save final model
#        self.saveModel(counter) 
#        
##    #build Q function - resuses discriminator for all layers except final layer
##    def discrQ(self, image, y):
##        with tf.variable_scope('discriminator') as scope:
##            scope.reuse_variables()
##            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
##            x = conv_cond_concat(image, yb)
##
##            h0 = lrelu(conv2d(x, self.df_dim, name='d_h0_conv'))
##            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv'), train=False))
##            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv'), train=False))
##            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv'), train=False))
##            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
##            #h4 is logits
##            return tf.nn.sigmoid(h4), h4
##            
##            
