# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def moving_average(a,w=10):
    if len(a)<w:
        return a[:]
    return [val if idx<w else sum(a[(idx-w):idx])/w for idx,val in enumerate(a)]

train_X=np.linspace(-1,1,100)
train_Y=2*train_X+np.random.randn(*train_X.shape)*0.3

#显示模拟数据点
plt.plot(train_X,train_Y,'ro',label='Original data')
plt.legend()
plt.show()


#重置图
tf.reset_default_graph()


#创建模型
#占位符
X=tf.placeholder("float")
Y=tf.placeholder("float")

#模型参数
W=tf.Variable(tf.random_normal([1]),name="weight")
b=tf.Variable(tf.zeros([1]),name="bias")
#前向结构
z=tf.multiply(X,W)+b
tf.summary.histogram('z',z)

#反向优化
cost=tf.reduce_mean(tf.square(Y-z))
tf.summary.scalar('loss_function',cost)

learning_rate=0.01
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


#训练模型

#初始化所有变量
init=tf.global_variables_initializer()
#定义参数
training_epochs=20
display_step=2

#生产saver
saver=tf.train.Saver(max_to_keep=1)
#生产模型的路径
savedir="log/"

#训练模型可视化

#启动session
with tf.Session() as sess:
    sess.run(init)
    
    merged_summary_op=tf.summary.merge_all()
    #创建summary_writer,用于写文件
    summary_writer=tf.summary.FileWriter('log/mnist_with_summaries',sess.graph)
    
    
    plotdata={"batchsize":[],"loss":[]}
    #向模型中输入数据
    for epoch in range(training_epochs):
        for (x,y) in zip(train_X,train_Y):
            sess.run(optimizer,feed_dict={X:x,Y:y})
        summary_str=sess.run(merged_summary_op,feed_dict={X:x,Y:y})
        summary_writer.add_summary(summary_str,epoch)
        if epoch % display_step ==0:
            loss=sess.run(cost,feed_dict={X:train_X,Y:train_Y})
            print("Epoch: ",epoch+1,"cost=",loss,"W=",sess.run(W),"b=",sess.run(b))
            if not (loss=="NA"):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)
            saver.save(sess,savedir+"linermodel.cpkt",global_step=epoch)
    
    print("Finished!")
    
    
    
    
    print("cost=",sess.run(cost,feed_dict={X:train_X,Y:train_Y}),"W=",sess.run(W),"b=",sess.run(b))
    plt.plot(train_X,train_Y,'ro',label='Original data')
    plt.plot(train_X,sess.run(W)*train_X+sess.run(b),label='Fittedline')
    plt.legend()
    plt.show()
    
    plotdata["avgloss"]=moving_average(plotdata["loss"])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["batchsize"],plotdata["avgloss"],'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel('loss')
    plt.title('Minibathc run vs. Traing loss')
    
    plt.show()
    
    #使用模型
    #print("x=0.2 ,z=",sess.run(z,feed_dict={X:0.2}))

##第一种方法：获取检查点文件  
#load_epoch=18
#with tf.Session() as sess2:
#    sess2.run(tf.global_variables_initializer())
#    saver.restore(sess2,savedir+"linermodel.cpkt-"+str(load_epoch))
#    print("x=0.2, z=",sess2.run(z,feed_dict={X:0.2}))
#
##第二种方法：获取检查点文件
#with tf.Session() as sess2:
#    sess2.run(tf.global_variables_initializer())
#    kpt=tf.train.latest_checkpoint(savedir)
#    if kpt!=None:
#        saver.restore(sess2,kpt)
#    #saver.restore(sess2,savedir+"linermodel.cpkt-"+str(load_epoch))
#    print("x=0.2, z=",sess2.run(z,feed_dict={X:0.2}))
