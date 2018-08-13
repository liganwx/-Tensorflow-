# -*- coding: utf-8 -*-


import tensorflow as tf
tf.reset_default_graph()
global_step=tf.train.get_or_create_global_step()
step=tf.assign_add(global_step,1)

#设置检查点路径为log/checkpoints
with tf.train.MonitoredTrainingSession(checkpoint_dir='log/checkpoints',save_checkpoint_secs=2) as sess:
    print(sess.run([global_step]))
    while not sess.should_stop():
        i=sess.run(step)
        print(i)
