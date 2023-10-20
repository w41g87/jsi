import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import math
import tensorflow as tf
from tqdm.notebook import trange

# helpers
def po2(n):
    x = 1
    while 2**x < n:
        x += 1
    return 2**x

def zipWith(arr1, arr2, func, dim):
    if (len(arr1) != len(arr2)):
        raise Exception('zipWith: Array dimension mismatch')
    output = np.empty((len(arr1), dim))
    for i in range(len(arr1)):
        output[i] = func(arr1[i], arr2[i])

    return output

def pltSect(input, x, y, sx, sy):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(25, 10), dpi=50)
    
    heatMap_mag = sns.heatmap(np.abs(input[x:sx + x, y:sy + y]), ax = ax[0], linewidth = 0, annot = False, cmap = "viridis")
    heatMap_mag.set_title("Magnitude Plot")
    heatMap_mag.set_xticks(np.arange(0, sx, int(np.ceil(sx / 20))))
    heatMap_mag.set_xticklabels(np.arange(x, x + sx, int(np.ceil(sx / 20))))
    heatMap_mag.set_yticks(np.arange(0, sy, int(np.ceil(sy / 20))))
    heatMap_mag.set_yticklabels(np.arange(-y, -y - sy, int(-np.ceil(sy / 20))), rotation = 0)

    heatMap_ph = sns.heatmap(np.angle(input[x:sx + x, y:sy + y]), ax = ax[1], vmin = 0, vmax = np.pi * 2, linewidth = 0, annot = False, cmap = sns.color_palette("blend:#491C62,#8E2E71,#C43F66,#E3695C,#EDB181,#E3695C,#C43F66,#8E2E71,#491C62", as_cmap=True))
    heatMap_ph.set_title("Phase Plot")
    heatMap_ph.set_xticks(np.arange(0, sx, int(np.ceil(sx / 20))))
    heatMap_ph.set_xticklabels(np.arange(x, x + sx, int(np.ceil(sx / 20))))
    heatMap_ph.set_yticks(np.arange(0, sy, int(np.ceil(sy / 20))))
    heatMap_ph.set_yticklabels(np.arange(-y, -y - sy, int(-np.ceil(sy / 20))), rotation = 0)
    fig.canvas.layout.width = '100%'
    fig.canvas.layout.height = '100%'
    fig.canvas.layout.overflow = 'scroll'
    fig.canvas.layout.padding = '0px'
    fig.canvas.layout.margin = '0px'

    return fig

def mkMtrx (nodes, js, phis, js_nh, phis_nh, g, y_0, y_ex, w, partial={}):
    output = np.zeros([nodes * 2, nodes * 2], dtype = np.complex_)
    cpl_a = []#np.zeros(3, dtype = np.complex_)
    cpl_s = []#np.zeros(3, dtype = np.complex_)

    # Assemble coupling constants
    for i in range(len(js)):
        cpl_a.append( (js[i][0] * np.exp(1j * phis[i][0]) + js_nh[i][0] * np.exp(1j * phis_nh[i][0]), js[i][1] * np.exp(1j * phis[i][1]) + js_nh[i][1] * np.exp(1j * phis_nh[i][1])) )
        cpl_s.append( (js[i][0] * np.exp(1j * phis[i][0]) - js_nh[i][0] * np.exp(1j * phis_nh[i][0]), js[i][1] * np.exp(1j * phis[i][1]) - js_nh[i][1] * np.exp(1j * phis_nh[i][1])) )
        
    # Signal side
    for i in range(nodes):
        row = nodes - i - 1
        if 'g' not in partial or (partial['g'] and i in partial['g']) or not partial['g']:
            output[nodes + i][row] = 1j * g
        if 'y' not in partial or (partial['y'] and i in partial['y']) or not partial['y']:
            output[row][row] = (y_0(True, i) + y_ex(True, i)) / 2 - 1j * w

        for k in range(len(js)):
            if (i - k - 1 >= 0):
                output[row + k + 1][row] = 1j * cpl_a[k][0] / 2
            if (i + k + 1 < nodes):
                output[row - k - 1][row] = 1j * cpl_s[k][0].conj() / 2

    # Idler side
    for i in range(nodes):
        row = nodes + i
        if 'g' not in partial or (partial['g'] and i in partial['g']) or not partial['g']:
            output[nodes - i - 1][row] = -1j * g
        if 'y' not in partial or (partial['y'] and i in partial['y']) or not partial['y']:
            output[row][row] = (y_0(False, i) + y_ex(False, i)) / 2 - 1j * w

        for k in range(len(js)):
            if (i - k - 1 >= 0):
                output[row - k - 1][row] = -1j * cpl_s[k][1] / 2
            if (i + k + 1 < nodes):
                output[row + k + 1][row] = -1j * cpl_a[k][1].conj() / 2
    
    return output

def mask(nodes, isSig, isRight, n):
    output = np.zeros([nodes * 2, nodes * 2], dtype = np.uint32)
    for i in range(nodes):
        if isSig:
            row = nodes - i - 1
            if (isRight and i - n >= 0):
                output[row + n][row] = 1
            if (not isRight and i + n < nodes):
                output[row - n][row] = 1
        else:
            row = nodes + i
            if (not isRight and i - n >= 0):
                output[row - n][row] = 1
            if (isRight and i + n < nodes):
                output[row + n][row] = 1
    return output

def maskG(nodes, isSig):
    output = np.zeros([nodes * 2, nodes * 2], dtype = np.uint32)
    for i in range(nodes):
        if isSig:
            output[nodes + i][nodes - i - 1] = 1
        else:
            output[nodes - i - 1][nodes + i] = 1
    return output
# Eigenvalue calculation
def eigenRSpace (nodes, js, phis, js_nh, phis_nh, g, y_0, y_ex):
    h_eff = mkMtrx(nodes, js, phis, js_nh, phis_nh, g, y_0, y_ex, 0) * (-1j)
    return np.linalg.eig(h_eff)

def eigenKSpace (nodes, js, phis, js_nh, phis_nh, g, y_0, y_ex):
    # Since the coupling value is not constant, the eigenvalue in k-space cannot be trivially computed
    
    # Descrete Fourier
    y_dft = lambda isSig, delta_k: np.sum(list(map(lambda n: (y_0(isSig, n) + y_ex(isSig, n)) * np.exp(1j * n * delta_k), range(nodes)))) / nodes
    h_eff = np.zeros([nodes * 2, nodes * 2], dtype = np.complex_)
    cpl_a = []#np.zeros(3, dtype = np.complex_)
    cpl_s = []#np.zeros(3, dtype = np.complex_)
    
    # Assemble coupling constants
    for i in range(len(js)):
        cpl_a.append( (js[i][0] * np.exp(1j * phis[i]) + js_nh[i][0] * np.exp(1j * phis_nh[i]), js[i][0] * np.exp(1j * phis[i]) + js_nh[i][1] * np.exp(1j * phis_nh[i])) )
        cpl_s.append( (js[i][1] * np.exp(1j * phis[i]) - js_nh[i][0] * np.exp(1j * phis_nh[i]), js[i][1] * np.exp(1j * phis[i]) - js_nh[i][1] * np.exp(1j * phis_nh[i])) )

    # Signal
    for i in range(nodes):
        row = nodes - i - 1
        h_eff[nodes + i][row] = g
        for k in range(len(js)):
            h_eff[row][row] += 1/2 * cpl_a[k][0] * np.exp(1j * i * (k+1) * 2 * np.pi / nodes)
            h_eff[row][row] += 1/2 * cpl_s[k][0].conj() * np.exp(-1j * i * (k+1) * 2 * np.pi / nodes)

        for k in range(nodes):
            col = nodes - k - 1
            h_eff[row][col] += -1j / 2 * y_dft(True, 2 * np.pi / nodes * (i - k)) if np.abs(y_dft(True, 2 * np.pi / nodes * (i - k))) > 1e-13 else 0 # rounding error dampening

    # Idler
    for i in range(nodes):
        row = nodes + i
        h_eff[nodes - i - 1][row] = -g
        for k in range(len(js)):
            h_eff[row][row] += -1/2 * cpl_s[k][1] * np.exp(1j * i * (k+1) * 2 * np.pi / nodes)
            h_eff[row][row] += -1/2 * cpl_a[k][1].conj() * np.exp(-1j * i * (k+1) * 2 * np.pi / nodes)
            
        for k in range(nodes):
            col = nodes + k
            h_eff[col][row] += -1j / 2 * y_dft(False, 2 * np.pi / nodes * (i - k)) if np.abs(y_dft(False, 2 * np.pi / nodes * (i - k))) > 1e-13 else 0 # rounding error dampening
            
    return np.linalg.eig(h_eff)

def eigenKSpaceP (period, js, phis, js_nh, phis_nh, g, y_0, y_ex):
    cpl_a = []#np.zeros(3, dtype = np.complex_)
    cpl_s = []#np.zeros(3, dtype = np.complex_)
    # cpl_a = np.zeros(len(js), dtype = np.complex_)
    # cpl_s = np.zeros(len(js), dtype = np.complex_)

    # Assemble coupling constants
    for i in range(len(js)):
        cpl_a.append( (js[i][0] * np.exp(1j * phis[i]) + js_nh[i][0] * np.exp(1j * phis_nh[i]), js[i][0] * np.exp(1j * phis[i]) + js_nh[i][1] * np.exp(1j * phis_nh[i])) )
        cpl_s.append( (js[i][1] * np.exp(1j * phis[i]) - js_nh[i][0] * np.exp(1j * phis_nh[i]), js[i][1] * np.exp(1j * phis[i]) - js_nh[i][1] * np.exp(1j * phis_nh[i])) )
    
    def func(k):
        h_eff = np.zeros([period * 2, period * 2], dtype = np.complex_)
        # Signal
        for i in range(period):
            row = period - i - 1
            h_eff[period + i][row] = g
            h_eff[row][row] += -1j / 2 * (y_0(True, i) + y_ex(True, i))
            for j in range(len(js)):
                if (row + (j + 1) < period):
                    h_eff[row + j + 1][row] += cpl_a[j][0] / 2
                else:
                    h_eff[(row + j + 1) % period][row] += cpl_a[j][0] / 2 * np.exp(1j * k)
                if (row - (j + 1) >= 0):
                    h_eff[row - (j + 1)][row] += cpl_s[j][0].conj() / 2
                else:
                    h_eff[(row - (j + 1)) % period][row] += cpl_s[j][0].conj() / 2 * np.exp(-1j * k)
    
        # Idler
        for i in range(period):
            row = period + i
            h_eff[period - i - 1][row] = -g
            h_eff[row][row] += -1j / 2 * (y_0(False, i) + y_ex(False, i))
            for j in range(len(js)):
                if (i - (j + 1) >= 0):
                    h_eff[row - (j + 1)][row] += -1 * cpl_s[j][1] / 2
                else:
                    h_eff[period + (i - (j + 1)) % period][row] += -1 * cpl_s[j][1] / 2 * np.exp(1j * k)
                if (i + (j + 1) < period):
                    h_eff[row + j + 1][row] += -1 * cpl_a[j][1].conj() / 2
                else:
                    h_eff[period + (i + j + 1) % period][row] += -1 * cpl_a[j][1].conj() / 2 * np.exp(-1j * k)
                            
        return np.linalg.eig(h_eff)
    return func

def eigSort(input):
    nodes = int(len(input[1][0]) / 2)
    output = [max( (np.abs(v), i) for i, v in enumerate(a) )[1] - nodes for a in input[1]]
    return [x % (nodes / 2) if x > 0 else (-x - 1) % (nodes / 2) for x in output]

def jsi(nodes, js, phis, js_nh, phis_nh, g, y_0, y_ex, w1, w2, w3, w4, partial={}):
    output = np.zeros((nodes, nodes), dtype = np.complex_)

    if (np.abs(w1 + w2) < 1e-13 and np.abs(w3 + w4) < 1e-13):
        m1 = mkMtrx(nodes, js, phis, js_nh, phis_nh, g, y_0, y_ex, w1, partial)
        m2 = mkMtrx(nodes, js, phis, js_nh, phis_nh, g, y_0, y_ex, -1 * w2, partial)
        m3 = mkMtrx(nodes, js, phis, js_nh, phis_nh, g, y_0, y_ex, -1 * w3, partial)
        m4 = mkMtrx(nodes, js, phis, js_nh, phis_nh, g, y_0, y_ex, w4, partial)
        # print(m1)
        m = tf.constant(np.array([m1, m2, m3, m4]), shape=(4, nodes * 2, nodes * 2), dtype = tf.complex64)
        y_ex_S = tf.constant(np.array([y_ex(True, x) for x in range(nodes)]), shape=(nodes, ), dtype = tf.complex64)
        y_ex_I = tf.constant(np.array([y_ex(False, x) for x in range(nodes)]), shape=(1, nodes), dtype=tf.complex64)
        y_0_S = tf.constant(np.array([y_0(True, x) for x in range(nodes)]), shape=(nodes, ), dtype=tf.complex64)
        y_0_I = tf.constant(np.array([y_0(False, x) for x in range(nodes)]), shape=(1, nodes), dtype=tf.complex64)
        
        u = tf.linalg.inv(m)
        # print(m[0].numpy())
        # print(u[0].numpy())
        u1 = tf.math.conj(tf.reverse(u[0, nodes : 2 * nodes, 0 : nodes], [1]))
        u2 = u[1, nodes : 2 * nodes, nodes : 2 * nodes]
        u3 = tf.math.conj(u[2, nodes : 2 * nodes, nodes : 2 * nodes])
        u4 = tf.reverse(u[3, nodes : 2 * nodes, 0 : nodes], [1])
        # print(u1.numpy())
        # print(u2.numpy())
        # print(u3.numpy())
        # print(u4.numpy())
        u1u2 = tf.linalg.matmul(u1, tf.transpose(u2))
        u3u4 = tf.linalg.matmul(u4, tf.transpose(u3))
    
        output = tf.multiply(y_ex_S, tf.multiply(y_ex_I, tf.multiply(u1 - tf.multiply(y_0_I + y_ex_I, u1u2), u4 - tf.multiply(y_0_S + y_ex_S, u3u4)) )).numpy() * 4 * np.power(np.pi, 2)
        # print(output)
    return output

def jsi_backprop(nodes, target, EPOCHS, lr=1e-4, length=5, padding=0):
    nodes_t = nodes + 2 * padding
    losses = np.zeros(int(EPOCHS), dtype=np.float32)
    pi = tf.constant(np.pi, dtype=tf.complex64)
    y_s_mask = tf.constant(mask(nodes_t, True, True, 0), dtype = tf.int32)
    y_i_mask = tf.constant(mask(nodes_t, False, True, 0), dtype = tf.int32)
    g_s_mask = tf.constant(maskG(nodes_t, True), dtype = tf.int32)
    g_i_mask = tf.constant(maskG(nodes_t, False), dtype = tf.int32)
    mask_arr = []
    
    for i in range(length):
        mask_arr.append( [mask(nodes_t, True, True, i + 1), mask(nodes_t, True, False, i + 1), mask(nodes_t, False, True, i + 1), mask(nodes_t, False, False, i + 1)] )

    def pos_real_constraint(x):
        return tf.cast(tf.abs(tf.math.real(x)), tf.complex64)
    j_mask = tf.constant(mask_arr, dtype=tf.int32)
    js_s = tf.Variable(tf.ones((length, ), dtype=tf.complex64) / 10, name="coupling_s")
    js_i = tf.Variable(tf.ones((length, ), dtype=tf.complex64) / 10, name="coupling_i")
    g = tf.Variable(0.03, dtype=tf.complex64, name="g", constraint=pos_real_constraint)
    y_ex_s = tf.Variable(0.1, dtype=tf.complex64, name="y_ex_s", constraint=pos_real_constraint)
    y_ex_i = tf.Variable(0.5, dtype=tf.complex64, name="y_ex_i", constraint=pos_real_constraint)
    y_0_s = tf.Variable(0.3, dtype=tf.complex64, name="y_0_s", constraint=pos_real_constraint)
    y_0_i = tf.Variable(0.1, dtype=tf.complex64, name="y_0_i", constraint=pos_real_constraint)
    target_t = tf.constant(target, shape=(nodes, nodes), dtype=tf.complex64)

    optimizer = tf.keras.optimizers.Adam(lr)

    @tf.function(input_signature=(
        tf.TensorSpec(shape=tf.shape(target_t), dtype=tf.complex64),
        tf.TensorSpec(shape=tf.shape(target_t), dtype=tf.complex64)
    ))
    def loss_func(target, pred):
        # RMS
        # return tf.sqrt(tf.reduce_mean(tf.math.real(tf.math.conj(output - target_t) * (output - target_t))))

        # vector projection
        return tf.cast(
                1 - tf.abs( tf.tensordot(tf.math.conj(target), pred, axes=2) * tf.tensordot(tf.math.conj(pred), target, axes=2) )
                           / (tf.abs((tf.tensordot(tf.math.conj(pred),  pred, axes=2)) * tf.tensordot(tf.math.conj(target), target, axes=2)))
                , tf.float32)
    
    @tf.function(input_signature=(
        tf.TensorSpec(shape=tf.shape(js_s), dtype=tf.complex64, name="coupling_s"),
        tf.TensorSpec(shape=tf.shape(js_i), dtype=tf.complex64, name="coupling_i"),
        tf.TensorSpec(shape=tf.shape(g), dtype=tf.complex64, name="g"),
        tf.TensorSpec(shape=tf.shape(y_ex_s), dtype=tf.complex64, name="y_ex_s"),
        tf.TensorSpec(shape=tf.shape(y_ex_i), dtype=tf.complex64, name="y_ex_i"),
        tf.TensorSpec(shape=tf.shape(y_0_s), dtype=tf.complex64, name="y_0_s"),
        tf.TensorSpec(shape=tf.shape(y_0_i), dtype=tf.complex64, name="y_0_i")
    ))
    def model(js_s, js_i, g, y_ex_s, y_ex_i, y_0_s, y_0_i):
        m = tf.where(tf.equal(y_s_mask, 1), (y_ex_s + y_0_s) / 2, tf.cast(tf.zeros_like(y_s_mask), tf.complex64)) \
                + tf.where(tf.equal(y_i_mask, 1), (y_ex_i + y_0_i) / 2, tf.cast(tf.zeros_like(y_i_mask), tf.complex64)) \
                + tf.where(tf.equal(g_s_mask, 1), 1j * g, tf.cast(tf.zeros_like(g_s_mask), tf.complex64)) \
                + tf.where(tf.equal(g_i_mask, 1), -1j * g, tf.cast(tf.zeros_like(g_i_mask), tf.complex64))

        for i in tf.range(length):
            m = m + tf.where(tf.equal(j_mask[i, 0], 1), 1j * js_s[i] / 2, tf.cast(tf.zeros_like(j_mask[i,0]), tf.complex64)) \
                    + tf.where(tf.equal(j_mask[i, 1], 1), 1j * tf.math.conj(js_s[i]) / 2, tf.cast(tf.zeros_like(j_mask[i,1]), tf.complex64)) \
                    + tf.where(tf.equal(j_mask[i, 2], 1), -1j * tf.math.conj(js_i[i]) / 2, tf.cast(tf.zeros_like(j_mask[i,2]), tf.complex64)) \
                    + tf.where(tf.equal(j_mask[i, 3], 1), -1j * js_i[i] / 2, tf.cast(tf.zeros_like(j_mask[i,3]), tf.complex64))
        
        # print(m.numpy())
        u = tf.linalg.inv(m)
        
        # print(u.numpy())
        u1 = tf.math.conj(tf.reverse(u[nodes_t: 2 * nodes_t, 0 : nodes_t], [1]))
        u2 = u[nodes_t : 2 * nodes_t, nodes_t : 2 * nodes_t]
        u3 = tf.math.conj(u[nodes_t : 2 * nodes_t, nodes_t : 2 * nodes_t])
        u4 = tf.reverse(u[nodes_t : 2 * nodes_t, 0 : nodes_t], [1])

        u1u2 = tf.linalg.matmul(u1, tf.transpose(u2))
        u3u4 = tf.linalg.matmul(u4, tf.transpose(u3))

        y_ex_s_t = tf.fill([nodes_t], y_ex_s)
        y_ex_i_t = tf.fill([1, nodes_t], y_ex_i)
        y_0_s_t = tf.fill([nodes_t], y_0_s)
        y_0_i_t = tf.fill([1, nodes_t], y_0_i)
        
        return (y_ex_s_t * y_ex_i_t * ((u1 - (y_0_i_t + y_ex_i_t) * u1u2) * (u4 - (y_0_s_t + y_ex_s_t) * u3u4)) * tf.constant(4, dtype=tf.complex64) * pi * pi)[padding : nodes + padding, padding : nodes + padding]

    @tf.function
    def train_step(loss_func, model, target, js_s, js_i, g, y_ex_s, y_ex_i, y_0_s, y_0_i):
        with tf.GradientTape() as tape:
            pred = model(js_s, js_i, g, y_ex_s, y_ex_i, y_0_s, y_0_i)
            loss = loss_func(target, pred)
            gradients = tape.gradient(loss, [js_s, js_i, g, y_ex_s, y_ex_i, y_0_s, y_0_i])  # Compute gradients
            optimizer.apply_gradients(zip(gradients, [js_s, js_i, g, y_ex_s, y_ex_i, y_0_s, y_0_i]))  # Update weights
        return loss
        
    for j in trange(int(EPOCHS), desc="iterations"):
        losses[j] = train_step(loss_func, model, target_t, js_s, js_i, g, y_ex_s, y_ex_i, y_0_s, y_0_i)
        
    return (js_s.numpy(), js_i.numpy(), g.numpy(), y_ex_s.numpy(), y_ex_i.numpy(), y_0_s.numpy(), y_0_i.numpy(), losses)