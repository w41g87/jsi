import sys, os, datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import math
import tensorflow as tf
from pathlib import Path
from tqdm.autonotebook import trange

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
    try:
        get_ipython
        fig.canvas.layout.width = '100%'
        fig.canvas.layout.height = '100%'
        fig.canvas.layout.overflow = 'scroll'
        fig.canvas.layout.padding = '0px'
        fig.canvas.layout.margin = '0px'
    except:
        pass

    return fig

def pltCtst(target, approx, x, y, sx, sy):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(25, 10), dpi=50)
    
    heatMap_t = sns.heatmap(np.abs(target[x:sx + x, y:sy + y]), ax = ax[0], linewidth = 0, annot = False, cmap = "viridis")
    heatMap_t.set_title("Target")
    heatMap_t.set_xticks(np.arange(0, sx, int(np.ceil(sx / 20))))
    heatMap_t.set_xticklabels(np.arange(x, x + sx, int(np.ceil(sx / 20))))
    heatMap_t.set_yticks(np.arange(0, sy, int(np.ceil(sy / 20))))
    heatMap_t.set_yticklabels(np.arange(-y, -y - sy, int(-np.ceil(sy / 20))), rotation = 0)

    heatMap_a = sns.heatmap(np.abs(approx[x:sx + x, y:sy + y]), ax = ax[1], linewidth = 0, annot = False, cmap = "viridis")
    heatMap_a.set_title("Approximation")
    heatMap_a.set_xticks(np.arange(0, sx, int(np.ceil(sx / 20))))
    heatMap_a.set_xticklabels(np.arange(x, x + sx, int(np.ceil(sx / 20))))
    heatMap_a.set_yticks(np.arange(0, sy, int(np.ceil(sy / 20))))
    heatMap_a.set_yticklabels(np.arange(-y, -y - sy, int(-np.ceil(sy / 20))), rotation = 0)
    try:
        get_ipython
        fig.canvas.layout.width = '100%'
        fig.canvas.layout.height = '100%'
        fig.canvas.layout.overflow = 'scroll'
        fig.canvas.layout.padding = '0px'
        fig.canvas.layout.margin = '0px'
    except:
        pass

    return fig

def data2HM(data):
    nodes = data['nodes']
    padding = data['padding']
    # js_p = list(zip(np.abs(data['js_s']), np.abs(data['js_i'])))
    # phis_p = list(zip(np.angle(data['js_s']), np.angle(data['js_i'])))
    js_p = list(zip(np.abs(data['js_s']), np.abs(data['js_s'])))
    phis_p = list(zip(np.angle(data['js_s']), np.angle(data['js_s'])))
    js_nh_p = list(zip(np.zeros(len(js_p)), np.zeros(len(js_p))))
    phis_nh_p = list(zip(np.zeros(len(js_p)), np.zeros(len(js_p))))
    g_p = data['g']
    
    def y_0_p(isSig, n):
        if isSig:
            return np.real(data['y_0_s'][0, n])
        else:
            return np.real(data['y_0_i'][n, 0])
    def y_ex_p(isSig, n):
        if isSig:
            return np.real(data['y_ex_s'][0, n])
        else:
            return np.real(data['y_ex_i'][n, 0])
            
    return pltSect(jsi(nodes + padding * 2, js_p, phis_p, js_nh_p, phis_nh_p, g_p, y_0_p, y_ex_p, 0, 0, 0, 0), padding, padding, nodes, nodes)

def mkMtrx (nodes, js, phis, js_nh, phis_nh, g, y_0, y_ex, w, partial={}):
    output = np.zeros([nodes * 2, nodes * 2], dtype = np.complex64)
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

def maskr(nodes, n_ring, n):
    output = np.zeros([nodes * 2 * n_ring, nodes * 2 * n_ring], dtype = np.uint32)
    pad = n * 2 * nodes
    for i in range(nodes):
        
        if n == 0:
            output[i][nodes * 2 - 1 - i] = 1
            output[nodes + i][nodes - i - 1] = -1
        else:
            output[pad - nodes * 2 + i][pad + i] = -1
            output[pad - nodes + i][pad + nodes + i] = 1
            # hermitian
            output[pad + i][pad - nodes * 2 + i] = -1
            output[pad + nodes + i][pad - nodes + i] = 1
            # non-hermitian?
            # output[pad + i][pad - nodes * 2 + i] = 1
            # output[pad + nodes + i][pad - nodes + i] = -1

        output[pad + i][pad + i] = 2
        output[pad + nodes + i][pad + nodes + i] = -2
        
        for j in [x + 1 for x in range(nodes - 1)]:
            if i + j < nodes:
                output[pad + i][pad + i + j] = 2 * j + 1
                output[pad + nodes + i][pad + nodes + i + j] = -2 * j - 1
            if i - j >= 0:
                output[pad + i][pad + i - j] = 2 * j + 2
                output[pad + nodes + i][pad + nodes + i - j] = -2 * j - 2
    return output

def mask2(nodes, n_ring, n):
    output = np.zeros([nodes * 2 * (n_ring + 1), nodes * 2 * (n_ring + 1)], dtype = np.uint32)
    pad = 0 if n == n_ring else n + 1 * 2 * nodes
    for i in range(nodes):
        output[pad + i][pad + i] = -1
        output[pad + nodes + i][pad + nodes + i] = 1
        if pad > 0:
            output[pad - nodes * 2 + i][pad + i] = -2
            output[pad - nodes + i][pad + nodes + i] = 2
            output[pad + i][pad - nodes * 2 + i] = -3
            output[pad + nodes + i][pad - nodes + i] = 3
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

def jsi(nodes, js, phis, js_nh, phis_nh, g, y_0, y_ex, w1, w2, w3, w4):
    warnings.warn("DEPRECATED: please use the jsi_backprop() with train=False option")
    output = np.zeros((nodes, nodes), dtype = np.complex64)

    if (np.abs(w1 + w2) < 1e-13 and np.abs(w3 + w4) < 1e-13):
        m1 = mkMtrx(nodes, js, phis, js_nh, phis_nh, g, y_0, y_ex, w1, partial)
        m2 = mkMtrx(nodes, js, phis, js_nh, phis_nh, g, y_0, y_ex, -1 * w2, partial)
        m3 = mkMtrx(nodes, js, phis, js_nh, phis_nh, g, y_0, y_ex, -1 * w3, partial)
        m4 = mkMtrx(nodes, js, phis, js_nh, phis_nh, g, y_0, y_ex, w4, partial)
        # print(m1)
        m = tf.constant(np.array([m1, m2, m3, m4]), shape=(4, nodes * 2, nodes * 2), dtype = tf.complex64)
        y_ex_S = tf.constant(np.array([y_ex(True, x) for x in range(nodes)]), shape=(nodes, 1), dtype = tf.complex64)
        y_ex_I = tf.constant(np.array([y_ex(False, x) for x in range(nodes)]), shape=(1, nodes), dtype=tf.complex64)
        y_0_S = tf.constant(np.array([y_0(True, x) for x in range(nodes)]), shape=(nodes, 1), dtype=tf.complex64)
        y_0_I = tf.constant(np.array([y_0(False, x) for x in range(nodes)]), shape=(1, nodes), dtype=tf.complex64)
        
        u = tf.linalg.inv(m)
        # print(m[0].numpy())
        # print(u[0].numpy())
        u1 = tf.math.conj(tf.reverse(u[0, nodes : 2 * nodes, 0 : nodes], [0]))
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

def jsi_backprop(init, EPOCHS=None, lr=1e-4, train=True):
    if train and not EPOCHS:
        raise ValueError("training parameter \'EPOCHS\' not provided")
    
    nodes = init.get('nodes')
    padding = init.get('padding', 0)
    target = init.get('target', None)
    n_rings = init.get('n_rings', 1)
    orth_itr = init.get('orth_itr', 3)
    data = init.copy()
    nodes_t = nodes + 2 * padding
    length = nodes_t - 1
    losses = np.zeros(int(EPOCHS if EPOCHS else 1), dtype=np.float32)
    pi = tf.constant(np.pi, dtype=tf.complex64)
    zero = tf.constant(0, dtype=tf.complex64)
    one = tf.constant(1, dtype=tf.complex64)
    mask_arr1 = []
    mask_arr2 = []
    if n_rings < 1:
        raise ValueError("ring count cannot be less than 1, but got " + str(n_rings))
        
    for i in range(n_rings):
        mask_arr1.append( maskr(nodes_t, n_rings, i) )
        mask_arr2.append( mask2(nodes_t, n_rings, i) )

    mask_arr2.append( mask2(nodes_t, n_rings, n_rings) )
    @tf.function
    def pos_real_constraint(x):
        return tf.cast(tf.abs(tf.math.real(x)), tf.complex64)
        
    masks = tf.constant(mask_arr1, dtype=tf.int32)
    masks2 = tf.constant(mask_arr2, dtype=tf.int32)
    js = tf.Variable(init.get('js', np.ones((n_rings, length), dtype=np.complex64) / 10), name="coupling", dtype=tf.complex64)
    g = tf.Variable(init.get('g', 0.03), dtype=tf.complex64, name="g", constraint=pos_real_constraint)
    jr = tf.Variable(init.get('jr', np.ones((n_rings, nodes_t * 2), dtype=np.complex64) / 10), name="interring", dtype=tf.complex64, constraint=pos_real_constraint)
    y0s = tf.Variable(init.get('y0s', np.ones((n_rings, nodes_t * 2), dtype=np.complex64) / 10), name="loss", dtype=tf.complex64, constraint=pos_real_constraint)

    target_t = tf.zeros(shape=(nodes, nodes), dtype=tf.complex64)
    if train:
        target_t = tf.constant(target, shape=(nodes, nodes), dtype=tf.complex64)
    output = None
    
    optimizer = tf.keras.optimizers.Adam(lr)

    @tf.function(input_signature=(
        tf.TensorSpec(shape=tf.shape(target_t), dtype=tf.complex64),
    ))
    def loss_func(pred):
        # RMS
        # return tf.sqrt(tf.reduce_mean(tf.math.real(tf.math.conj(output - target_t) * (output - target_t))))

        # vector projection
        return tf.cast(
                1 - tf.abs( tf.tensordot(tf.math.conj(target_t), pred, axes=2) * tf.tensordot(tf.math.conj(pred), target_t, axes=2) )
                           / (tf.abs((tf.tensordot(tf.math.conj(pred),  pred, axes=2)) * tf.tensordot(tf.math.conj(target_t), target_t, axes=2)))
                , tf.float32)
    
    @tf.function(input_signature=(
        tf.TensorSpec(shape=tf.shape(js), dtype=tf.complex64, name="coupling"),
        tf.TensorSpec(shape=tf.shape(jr), dtype=tf.complex64, name="interring"),
        tf.TensorSpec(shape=tf.shape(g), dtype=tf.complex64, name="g"),
        tf.TensorSpec(shape=tf.shape(y0s), dtype=tf.complex64, name="loss")
    ))
    def model(js, jr, g, y0s):
        m = tf.linalg.diag(tf.reshape(y0s / 2, [-1]))
        a = tf.where(tf.equal(masks2[-1], 1), one, zero) \
            + tf.where(tf.equal(masks2[-1], -1), one, zero)
        b = tf.zeros(((n_rings + 1) * 2 * nodes_t, (n_rings + 1) * 2 * nodes_t), dtype=tf.complex64)
        for i in tf.range(n_rings):
            # a += tf.where(tf.equal(masks2[i], -1), -1j * tf.math.sqrt(tf.reshape(y0s, [-1])), 0) \
            #     + tf.where(tf.equal(masks2[i], 1), 1j * tf.math.sqrt(tf.reshape(y0s, [-1])), 0)
            a += -1j * tf.math.sqrt(tf.where(tf.equal(masks2[i], 1), tf.tile(y0s[i], [n_rings + 1]), 0)) \
                + 1j * tf.math.sqrt(tf.where(tf.equal(masks2[i], -1), tf.tile(y0s[i], [n_rings + 1]), 0))

            if i == 0: 
                # first ring
                m += tf.where(tf.equal(masks[0], 1), -1j * g, 0) \
                    + tf.where(tf.equal(masks[0], -1), 1j * g, 0) \
                    + tf.where(tf.equal(masks[0], 2), tf.tile((jr[0] + (jr[1] if n_rings > 1 else zero)) / 2, [n_rings]), 0) \
                    + tf.where(tf.equal(masks[0], -2), tf.tile((jr[0] + (jr[1] if n_rings > 1 else zero)) / 2, [n_rings]), 0)
                b += tf.where(tf.equal(masks2[i], -3), tf.tile(1j * tf.math.sqrt(jr[i]), [n_rings + 1]), 0) \
                    + tf.where(tf.equal(masks2[i], 3), tf.tile(-1j * tf.math.sqrt(jr[i]), [n_rings + 1]), 0) # idler conj
                # TODO: contruct b for i == 0
            
            else:
                m += tf.where(tf.equal(masks[i], 1), tf.tile(1j * jr[i] / 2, [n_rings]), 0) \
                    + tf.where(tf.equal(masks[i], -1), tf.tile(-1j * jr[i] / 2, [n_rings]), 0) \
                    + tf.where(tf.equal(masks[i], 2), tf.tile((jr[i] + (jr[i + 1] if n_rings > (i + 1) else zero)) / 2, [n_rings]), 0) \
                    + tf.where(tf.equal(masks[i], -2), tf.tile((jr[i] + (jr[i + 1] if n_rings > (i + 1) else zero)) / 2, [n_rings]), 0)
        
                b += tf.where(tf.equal(masks2[i], 2), tf.tile(1j * jr[i] / 2, [n_rings + 1]), 0) \
                    + tf.where(tf.equal(masks2[i], -2), tf.tile(-1j * jr[i] / 2, [n_rings + 1]), 0) \
                    + tf.where(tf.equal(masks2[i], 3), tf.tile(1j * jr[i] / 2, [n_rings + 1]), 0) \
                    + tf.where(tf.equal(masks2[i], -3), tf.tile(-1j * jr[i] / 2, [n_rings + 1]), 0)
                # b += tf.where(tf.equal(masks2[i], 2), 1j * jr[i - 1] / 2, 0) \
                #     + tf.where(tf.equal(masks2[i], -2), -1j * jr[i - 1] / 2, 0) \
                #     + tf.where(tf.equal(masks2[i], 3), 1j * jr[i] / 2, 0) \
                #     + tf.where(tf.equal(masks2[i], -3), -1j * jr[i] / 2, 0)
            for j in tf.range(length):  
                m += tf.where(tf.equal(masks[i], j * 2 + 3), 1j * js[i][j] / 2, 0) \
                    + tf.where(tf.equal(masks[i], j * 2 + 4), 1j * tf.math.conj(js[i][j]) / 2, 0) \
                    + tf.where(tf.equal(masks[i], -j * 2 - 3), -1j * tf.math.conj(js[i][j])/ 2, 0) \
                    + tf.where(tf.equal(masks[i], -j * 2 - 4), -1j * js[i][j] / 2, 0)
        # print(m)

        for i in tf.range(orth_itr):
            a = a + tf.matmul(b, a)
            b = tf.matmul(b, b)
            
        u = tf.matmul(tf.linalg.inv(m), a[nodes_t * 2 : nodes_t * 2 * (n_rings + 1)])
        # print(tf.linalg.eigvals(m).numpy())
        # print(tf.linalg.eigvals(u[0 : nodes_t * 2, 0: 2 * nodes_t]).numpy())
        u4 = tf.reverse(u[0 : nodes_t, nodes_t: 2 * nodes_t], [0])
        u1 = tf.math.conj(u4)

        u4_p = tf.reverse(tf.concat([u[0 : nodes_t, (i * 2 + 1) * nodes_t : (i * 2 + 2) * nodes_t] for i in range(n_rings + 1)], axis=1), [0])
        u2_p = tf.concat([u[nodes_t : 2 * nodes_t, (i * 2 + 1) * nodes_t : (i * 2 + 2) * nodes_t] for i in range(n_rings + 1)], axis=1)
        u3_p = tf.math.conj(u2_p)
        u1_p = tf.math.conj(u4_p)

        u1u2_p = tf.linalg.matmul(u1_p, tf.transpose(u2_p))
        u3u4_p = tf.linalg.matmul(u4_p, tf.transpose(u3_p))

        return (tf.reshape(tf.reverse(jr[0][0 : nodes_t], [0]), (nodes_t, 1)) * (u1 - 1j * tf.math.sqrt(jr[0][nodes_t : nodes_t * 2]) * u1u2_p) * (u4 + 1j * tf.math.sqrt(jr[0][nodes_t : nodes_t * 2]) * u3u4_p) * tf.constant(4, dtype=tf.complex64) * pi * pi)[padding : nodes + padding, padding : nodes + padding]

    @tf.function
    def train_step(loss_func, model, js, jr, g, y0s):
        with tf.GradientTape() as tape:
            pred = model(js, jr, g, y0s)
            loss = loss_func(pred)
            gradients = tape.gradient(loss, [js, jr, g, y0s])  # Compute gradients
            optimizer.apply_gradients(zip(gradients, [js, jr, g, y0s]))  # Update weights
        return loss
        
    try:
        tf.profiler.experimental.stop()
    except:
        pass
    try:
        # tf.profiler.experimental.start('log/' + datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S') + '.log')
        if train:
            for j in trange(int(EPOCHS), desc="iterations"):
                loss = train_step(loss_func, model, js, jr, g, y0s)
                losses[j] = loss.numpy()
        else:
            losses = [init.get('loss', 0), ]

        # tf.profiler.experimental.stop()
    except KeyboardInterrupt:
        # tf.profiler.experimental.stop()
        data['js'] = js.numpy()
        data['jr'] = jr.numpy()
        data['g'] = g.numpy()
        data['y0s'] = y0s.numpy()
        data['loss'] = losses[-1]
        np.savez(Path('./_chkpt.npz').resolve(), **data)
        print("Interrupted. Progress is saved at _chkpt.npz")
        try:
            data['int'] = True
            return (data, losses, model(js, jr, g, y0s).numpy())
        except:
            try:
                sys.exit(130)
            except SystemExit:
                os._exit(130)

    data['js'] = js.numpy()
    data['jr'] = jr.numpy()
    data['g'] = g.numpy()
    data['y0s'] = y0s.numpy()
    data['loss'] = losses[-1]
    return (data, losses, model(js, jr, g, y0s).numpy())