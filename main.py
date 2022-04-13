import os
import numpy as np

from myBaseModels import mnistDCGan,DCGan
from myCondGAN import gestCondDCGan
from utils import pp, visualize,visGestIKCondGAN, show_all_variables

import tensorflow as tf
print('Using Tensorflow version : {}'.format(tf.__version__))

import argparse

#reset the computational graph on entry
tf.reset_default_graph()
#reset flags variable
tf.app.flags.FLAGS = tf.flags._FlagValues()
tf.app.flags._global_parser = argparse.ArgumentParser()
#set up flags - shortcut for argparse in tensorflow


#for gestIKCond2 dataset, use
#input_height = 200
#input_fname_pattern = "8.png"
#--resize
#--resize_height=64
#output_height = 64
#batch_size=128
#don't forget train flag

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "gestIKCond", "The name of dataset [celebA, mnist, gestIKCond, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")

flags.DEFINE_boolean("resize", False, "True to resize images if size != specified input_height/width, False for crop [False]")
flags.DEFINE_integer("resize_height", 64, "The size the input images should be resized to, if resize is specified [64]")
flags.DEFINE_integer("resize_width", None, "The size the input images should be resized to, if resize is specified. If None, same value as resize_height [None]")

flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")

FLAGS = flags.FLAGS

#settings for current run
#--dataset gestIKCond2 
#--input_height=200 
#--input_fname_pattern="*.png" 
#--resize 
#--resize_height=64 
#--output_height=64 
#--batch_size=128 



def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height


    if FLAGS.resize_width is None:
        FLAGS.resize_width = FLAGS.resize_height

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth=True
    #output destinations for samples during training
    sample_dir = os.path.join(FLAGS.sample_dir,FLAGS.dataset)
    # number of samples to save during training
    sample_num = FLAGS.batch_size
  
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)
    print('Saving samples in directory %s' % sample_dir)
    with tf.Session(config=run_config) as sess:
        if FLAGS.dataset == 'mnist':
            #slightly less deep DCGan with conditional capabilities to handle mnist
            dcgan = mnistDCGan(
                    sess,
                    y_dim=10,
                    flags=FLAGS,
                    sample_num=sample_num,
                    sample_dir=sample_dir)
        elif FLAGS.dataset == 'gestIKCond':
            #DCGan with conditional capabilities
            dcgan = gestCondDCGan(
                    sess,
                    flags=FLAGS,
                    sample_num=sample_num,
                    y_dim=36,   #26 letters (onehot) + 10 positions within sequence (not onehot) idxs 0-10
                    sample_dir=sample_dir)        
        elif FLAGS.dataset == 'gestIKCond2':
            #DCGan with conditional capabilities
            dcgan = gestCondDCGan(
                    sess,
                    flags=FLAGS,
                    sample_num=sample_num,
                    y_dim=44,   #26 letters (onehot) + 10 seq pos + 8 floats for camera translation, zoom, and orientation
                    sample_dir=sample_dir)        
        else:
            #base DCGan, with no conditional capabilities
            dcgan = DCGan(
                    sess,
                    flags=FLAGS,
                    sample_num=sample_num,
                    sample_dir=sample_dir)


        show_all_variables()
    
        if FLAGS.train:
            dcgan.train(FLAGS)
        else:
            if not dcgan.checkForModelAndLoad()[0]:
                raise Exception("[!] Train a model first, then run test mode")
#            if not dcgan.load(FLAGS.checkpoint_dir)[0]:
#                raise Exception("[!] Train a model first, then run test mode")
    
    
        # to_json("./web/js/layers.js", [dcgan.h0_w, dcgan.h0_b, dcgan.g_bn0],
        #                 [dcgan.h1_w, dcgan.h1_b, dcgan.g_bn1],
        #                 [dcgan.h2_w, dcgan.h2_b, dcgan.g_bn2],
        #                 [dcgan.h3_w, dcgan.h3_b, dcgan.g_bn3],
        #                 [dcgan.h4_w, dcgan.h4_b, None])
        
        #Evaluate model      
        if FLAGS.dataset == 'gestIKCond':
            #only gesture sequences, camera static
            option = 1
            visGestIKCondGAN(sess, dcgan, option)
        elif FLAGS.dataset == 'gestIKCond2':
            #gesture sequences with moving camera
#            option = 1
#            visGestIKCondGAN(sess, dcgan, option, useCamVals=True)
#            option = 2
#            visGestIKCondGAN(sess, dcgan, option, useCamVals=True)
#            option = 3
#            visGestIKCondGAN(sess, dcgan, option, useCamVals=True)
            option = 4
            visGestIKCondGAN(sess, dcgan, option, useCamVals=True)
        else:   
            # Below is codes for visualization of base DCGan and mnist gan
            OPTION = 1
            visualize(sess, dcgan, FLAGS, OPTION)
            
        #end with session

if __name__ == '__main__':
  tf.app.run()
