import tensorflow as tf
import load_data
import time
import os
import numpy as np
import u_net
import cv2

tf_v1 = tf.compat.v1
tf_v1.disable_v2_behavior()


# ==== setting ==== #
Height = 224
Width = 224
batch_size = 16
test_batch_size = 10
learning_rate = 1e-3
saving_iter = 3 # save per 1epoch
Max_iter = 1000
# ================= #

# ==== load data ==== #
print('== data loading ==')
color_dict = {0: (0, 0, 0), 21: (192, 224, 224), 1: (0, 0, 128), 2: (128, 128, 192), 3: (128, 64, 0), 4: (0, 0, 192), 5: (128, 0, 64), 6: (0, 128, 128), 7: (128, 0, 128), 8: (128, 0, 0), 9: (0, 128, 192), 10: (0, 192, 128), 11: (128, 128, 64), 12: (128, 0, 192), 13: (0, 128, 64), 14: (128, 128, 128), 15: (0, 128, 0), 16: (0, 0, 64), 17: (0, 192, 0), 18: (128, 128, 0), 19: (0, 64, 0), 20: (0, 64, 128)}
X_train, Y_train, X_val, Y_val = load_data.load_data(Width, Height)
print('== data loaded ==')
# =================== #

# ==== don't change ==== #
train_data_size = len(X_train)
test_data_size = len(X_val)
num_class = 20+2 # cls + ground : 0 + border : 21
saving_iter *= int((train_data_size//batch_size))
Max_iter *= saving_iter # max epochs
# =================== #

# ---- model location ---- #
model_path = './model/u_net'
if not os.path.exists(model_path):
    os.makedirs(model_path)
# ------------------------ #

# ==== model restore ==== #
restore = False

restore_point = 17  # restore epoch
Checkpoint = model_path + '/cVG epoch ' + str(restore_point) + '/'
WeightName = Checkpoint + 'Train_' + str(restore_point) + '.meta'

if not restore:
    restore_point = 0
# ======================== #

# ==== variables ==== #
X = tf_v1.placeholder(tf.float32, [None, Height, Width, 3])
C = tf_v1.placeholder(tf.int32, [None,Height,Width])
LR = tf_v1.placeholder(tf.float32)
PHASE = tf_v1.placeholder(tf.bool)
# =================== #

#            model                #
pred = u_net.U_NET(X, num_class,name='fcn_net/', reuse=False)

onehot = tf.one_hot(C, num_class, axis=-1)

# Loss_C = tf.reduce_mean(tf.divide(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=onehot), axis=(1, 2)), not_boundary_count))
Loss_C = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=onehot)) # [:,:,:,:-1] -> exclude boundary loss


# --------------------- variable & optimizer
var = [v for v in tf_v1.global_variables() if v.name.startswith('fcn_net/')]
update_ops = tf_v1.get_collection(tf_v1.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    # optimizer = tf_v1.train.GradientDescentOptimizer(learning_rate=LR).minimize(Loss_C, var_list=var)
    optimizer = tf_v1.train.AdamOptimizer(learning_rate=LR).minimize(Loss_C, var_list=var)


# --------- Run

sess = tf_v1.Session()

init = tf_v1.global_variables_initializer()
sess.run(init)
saver = tf_v1.train.Saver(max_to_keep=10)

if restore:
    print('Weight Restoring.....')
    Restore = tf_v1.train.import_meta_graph(WeightName)
    Restore.restore(sess, tf.train.latest_checkpoint(Checkpoint))
    print('Weight Restoring Finish!')

# ---- train ---- #
start_time = time.time()
init_lr = learning_rate


for iter_count in range(restore_point * saving_iter + 1, saving_iter * Max_iter+1):

    img, cls = load_data.make_train_batch(X_train, Y_train, batch_size, Width, Height)
    _, LossC_val = sess.run([optimizer, Loss_C], feed_dict={X: img, C: cls, PHASE: True, LR: learning_rate})

    if iter_count % 1 == 0:
        consume_time = time.time() - start_time
        print('%d    Loss = %.5f   LR = %.5f   time = %.4f    ' % (iter_count, LossC_val, learning_rate, consume_time))
        start_time = time.time()

    if iter_count % (saving_iter) == 0:
        print('SAVING MODEL')
        Temp = model_path + '/cVG epoch %d/' % int((iter_count // saving_iter))

        if not os.path.exists(Temp):
            os.makedirs(Temp)

        SaveName = Temp + 'Train_%s' % int((iter_count // saving_iter))
        saver.save(sess, SaveName)
        print('SAVING MODEL Finish')

        # ---- test ---- #
        test_pixel_acc = 0
        boundary_count = 0
        for it in range(0, test_data_size, test_batch_size):
            if it % 100 == 0:
                print('%d  /  %d ' % (it, test_data_size))

            img, cls = load_data.make_test_batch(X_val, Y_val, test_batch_size, it, Width, Height)
            np_pred = sess.run(pred, feed_dict={X: img, PHASE: False})
            boundary_count += np.sum(cls == 21)
            np_pred = np.argmax(np_pred[:,:,:,:-1],axis=-1)
            test_pixel_acc += np.sum(np_pred == cls)

            temp = np_pred[0]
            img_bgr = np.zeros((Height,Width,3))
            cls_bgr = np.zeros((Height,Width,3))

            if len(np.unique(temp)) != 1: # model 이 0만 predict 하면 이미지를 출력하지 않음
                for w in range(Width):
                    for h in range(Height):
                        img_bgr[w,h] = color_dict[temp[w,h]]
                        cls_bgr[w,h] = color_dict[cls[0][w,h]]

                cv2.imwrite(f'./img/test/{it}_p.png',img_bgr)
                cv2.imwrite(f'./img/test/{it}_c.png',cls_bgr)
        test_pixel_acc = test_pixel_acc  / (Height * Width * train_data_size - boundary_count) * 100

        # train acc
        train_pixel_acc = 0
        boundary_count = 0
        for it in range(0, train_data_size, test_batch_size):
            if it % 100 == 0:
                print('%d  /  %d ' % (it, train_data_size))

            img, cls = load_data.make_test_batch(X_train, Y_train, test_batch_size, it, Width, Height)
            np_pred = sess.run(pred, feed_dict={X: img, PHASE: False})
            boundary_count += np.sum(cls == 21)
            np_pred = np.argmax(np_pred[:,:,:,:-1],axis=-1)
            train_pixel_acc += np.sum(np_pred == cls)

            temp = np_pred[0]
            img_bgr = np.zeros((Height,Width,3))
            cls_bgr = np.zeros((Height,Width,3))

            if len(np.unique(temp)) != 1: # model 이 0만 predict 하면 이미지를 출력하지 않음
                for w in range(Width):
                    for h in range(Height):
                        img_bgr[w,h] = color_dict[temp[w,h]]
                        cls_bgr[w,h] = color_dict[cls[0][w,h]]

                cv2.imwrite(f'./img/train/{it}_p.png',img_bgr)
                cv2.imwrite(f'./img/train/{it}_c.png',cls_bgr)

        train_pixel_acc = train_pixel_acc  / (Height * Width * train_data_size - boundary_count) * 100

        f = open('accuracy.txt', 'a+')
        f.write("epoch : %d  /  " % (iter_count/saving_iter))
        f.write('train_Pixel_Acc : %f  /  ' % (train_pixel_acc))
        f.write("test_Pixel_Acc : %f\n" % (test_pixel_acc))
        f.close()