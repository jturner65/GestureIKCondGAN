import os
import scipy.misc
import numpy as np

from myModel import mnistDCGan,baseDCGan,gestCondDCGan
#from model import DCGAN
from utils import pp, visualize, to_json, show_all_variables

import tensorflow as tf

import argparse

#reset the computational graph on entry
tf.reset_default_graph()
#reset flags variable
tf.app.flags.FLAGS = tf.flags._FlagValues()
tf.app.flags._global_parser = argparse.ArgumentParser()
#set up flags - shortcut for argparse in tensorflow
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
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, gestIKCond, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")

flags.DEFINE_boolean("resize", False, "True to resize images if size != specified input_height/width, False for crop [False]")

flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")

FLAGS = flags.FLAGS

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if FLAGS.input_width is None:
    FLAGS.input_width = FLAGS.input_height
  if FLAGS.output_width is None:
    FLAGS.output_width = FLAGS.output_height

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True
  #output desitnation for samples during training
  sample_dir = os.path.join(FLAGS.sample_dir,FLAGS.dataset)
  # number of samples to save during training
  sample_num = FLAGS.batch_size
  if not os.path.exists(sample_dir):
    os.mkdir(sample_dir)
  print('Saving samples in directory %s' % sample_dir)
  with tf.Session(config=run_config) as sess:
    if FLAGS.dataset == 'mnist':
      dcgan = mnistDCGan(
          sess,
          y_dim=10,
          flags=FLAGS,
          sample_num=sample_num,
          sample_dir=sample_dir)
    elif FLAGS.dataset == 'gestIKCond':
      dcgan = gestCondDCGan(
          sess,
          flags=FLAGS,
          sample_num=sample_num,
          y_dim=36,   #26 letters (onehot) + 10 positions within sequence (not onehot) idxs 0-10
          sample_dir=sample_dir)        
    else:
      dcgan = baseDCGan(
          sess,
          flags=FLAGS,
          sample_num=sample_num,
          sample_dir=sample_dir)
#    if FLAGS.dataset == 'mnist':
#      dcgan = mnistDCGan(
#          sess,
#          input_width=FLAGS.input_width,
#          input_height=FLAGS.input_height,
#          output_width=FLAGS.output_width,
#          output_height=FLAGS.output_height,
#          batch_size=FLAGS.batch_size,
#          sample_num=FLAGS.batch_size,
#          y_dim=10,
#          dataset_name=FLAGS.dataset,
#          input_fname_pattern=FLAGS.input_fname_pattern,
#          crop=FLAGS.crop,
#          checkpoint_dir=FLAGS.checkpoint_dir,
#          sample_dir=sample_dir)
#    elif FLAGS.dataset == 'gestIKCond':
#      dcgan = gestCondDCGan(
#          sess,
#          input_width=FLAGS.input_width,
#          input_height=FLAGS.input_height,
#          output_width=FLAGS.output_width,
#          output_height=FLAGS.output_height,
#          batch_size=FLAGS.batch_size,
#          sample_num=FLAGS.batch_size,
#          y_dim=36,   #26 letters (onehot) + 10 positions within sequence (not onehot) idxs 0-10
#          dataset_name=FLAGS.dataset,
#          input_fname_pattern=FLAGS.input_fname_pattern,
#          crop=FLAGS.crop,
#          checkpoint_dir=FLAGS.checkpoint_dir,
#          sample_dir=sample_dir)        
#    else:
#      dcgan = baseDCGan(
#          sess,
#          input_width=FLAGS.input_width,
#          input_height=FLAGS.input_height,
#          output_width=FLAGS.output_width,
#          output_height=FLAGS.output_height,
#          batch_size=FLAGS.batch_size,
#          sample_num=FLAGS.batch_size,
#          dataset_name=FLAGS.dataset,
#          input_fname_pattern=FLAGS.input_fname_pattern,
#          crop=FLAGS.crop,
#          checkpoint_dir=FLAGS.checkpoint_dir,
#          sample_dir=sample_dir)

    show_all_variables()

    if FLAGS.train:
      dcgan.train(FLAGS)
    else:
      if not dcgan.load(FLAGS.checkpoint_dir)[0]:
        raise Exception("[!] Train a model first, then run test mode")
      

    # to_json("./web/js/layers.js", [dcgan.h0_w, dcgan.h0_b, dcgan.g_bn0],
    #                 [dcgan.h1_w, dcgan.h1_b, dcgan.g_bn1],
    #                 [dcgan.h2_w, dcgan.h2_b, dcgan.g_bn2],
    #                 [dcgan.h3_w, dcgan.h3_b, dcgan.g_bn3],
    #                 [dcgan.h4_w, dcgan.h4_b, None])

    # Below is codes for visualization
    OPTION = 1
    visualize(sess, dcgan, FLAGS, OPTION)

if __name__ == '__main__':
  tf.app.run()