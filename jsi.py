import sys, os, datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import math
import tensorflow as tf
from tensorflow.keras import layers, models, losses
from keras.datasets import mnist
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

@tf.keras.saving.register_keras_serializable()
class JsiKernel:
    def __init__(self, nodes, padding=0, n_rings=1, length=None, orth_itr=5):
        self.nodes = nodes
        self.padding = padding
        self.nodes_t = nodes + 2 * padding
        if n_rings < 1:
            raise ValueError("ring count cannot be less than 1, but got " + str(n_rings))
        self.n_rings = n_rings
        self.length = length if length else nodes_t - 1
        self.orth_itr = orth_itr

        mask_arr1 = []
        mask_arr2 = []
        for i in range(self.n_rings):
            mask_arr1.append( self.maskr(self.nodes_t, self.n_rings, i) )
            mask_arr2.append( self.mask2(self.nodes_t, self.n_rings, i) )
            
        mask_arr2.append( self.mask2(self.nodes_t, self.n_rings, self.n_rings) )
        self.masks = tf.constant(mask_arr1, dtype=tf.int32)
        self.masks2 = tf.constant(mask_arr2, dtype=tf.int32)

        self.pi = tf.constant(np.pi, dtype=tf.complex64)
        self.zero = tf.constant(0, dtype=tf.complex64)
        self.one = tf.constant(1, dtype=tf.complex64)
        
    def get_config(self):
        return {'nodes': self.nodes, 
                'padding': self.padding, 
                'n_rings': self.n_rings, 
                'length': self.length, 
                'orth_itr': self.orth_itr
               }

    def maskr(self, nodes, n_ring, n):
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
    
    def mask2(self, nodes, n_ring, n):
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
    
    @tf.function
    def call_flat(self, pred):
        js = tf.cast(tf.reshape(pred[0:self.n_rings * self.length], (self.n_rings, self.length)), dtype=tf.complex64) * tf.exp(1j * tf.cast(tf.reshape(pred[self.n_rings * self.length:self.n_rings * self.length * 2], (self.n_rings, self.length)), dtype=tf.complex64))
        jr = tf.cast(tf.reshape(pred[self.n_rings*self.length*2 : self.n_rings*self.length*2+self.nodes_t*2*self.n_rings], (self.n_rings, self.nodes_t*2)), dtype=tf.complex64)
        g = tf.cast(pred[-1], dtype=tf.complex64)
        y0s = tf.cast(tf.reshape(pred[self.n_rings*self.length*2+self.nodes_t*2*self.n_rings:self.n_rings*self.length*2+self.nodes_t*4*self.n_rings], (self.n_rings, self.nodes_t*2)), dtype=tf.complex64)
        return self(js, jr, g, y0s)
        
    @tf.function
    # @tf.function(input_signature=(
    #     tf.TensorSpec(shape=tf.shape(js), dtype=tf.complex64, name="coupling"),
    #     tf.TensorSpec(shape=tf.shape(jr), dtype=tf.complex64, name="interring"),
    #     tf.TensorSpec(shape=tf.shape(g), dtype=tf.complex64, name="g"),
    #     tf.TensorSpec(shape=tf.shape(y0s), dtype=tf.complex64, name="loss")
    # ))
    def __call__(self, js, jr, g, y0s):
        m = tf.linalg.diag(tf.reshape(y0s / 2, [-1]))
        # a = tf.where(tf.equal(self.masks2[-1], 1), self.one, self.zero) \
        #     + tf.where(tf.equal(self.masks2[-1], -1), self.one, self.zero)
        # a = -1j * tf.math.sqrt(tf.where(tf.equal(self.masks2[-1], 1), tf.tile(jr[0], [self.n_rings + 1]), 0)) \
        #     + 1j * tf.math.sqrt(tf.where(tf.equal(self.masks2[-1], -1), tf.tile(jr[0], [self.n_rings + 1]), 0))
        # b = tf.zeros(((self.n_rings + 1) * 2 * self.nodes_t, (self.n_rings + 1) * 2 * self.nodes_t), dtype=tf.complex64)

        # jr_sqrt = tf.math.sqrt(jr)
        
        for i in tf.range(self.n_rings):
            # a += -1j * tf.math.sqrt(tf.where(tf.equal(self.masks2[i], 1), tf.tile(y0s[i], [self.n_rings + 1]), 0)) \
            #     + 1j * tf.math.sqrt(tf.where(tf.equal(self.masks2[i], -1), tf.tile(y0s[i], [self.n_rings + 1]), 0))
            # b += tf.where(tf.equal(self.masks2[i], 2), tf.tile(-1j * jr_sqrt[i], [self.n_rings + 1]), 0) \
            #     + tf.where(tf.equal(self.masks2[i], -2), tf.tile(1j * jr_sqrt[i], [self.n_rings + 1]), 0) \
            #     + tf.where(tf.equal(self.masks2[i], 3), tf.tile(-1j * jr_sqrt[i], [self.n_rings + 1]), 0) \
            #     + tf.where(tf.equal(self.masks2[i], -3), tf.tile(1j * jr_sqrt[i], [self.n_rings + 1]), 0)
            
            if i == 0: 
                # first ring
                m += tf.where(tf.equal(self.masks[0], 1), 1j * g, 0) \
                    + tf.where(tf.equal(self.masks[0], -1), -1j * g, 0) \
                    + tf.where(tf.equal(self.masks[0], 2), tf.tile(jr[0] / 2, [self.n_rings]), 0) \
                    + tf.where(tf.equal(self.masks[0], -2), tf.tile(jr[0] / 2, [self.n_rings]), 0)
                    # + tf.where(tf.equal(masks[0], 2), tf.tile((jr[0] + (jr[1] if n_rings > 1 else zero)) / 2, [n_rings]), 0) \
                    # + tf.where(tf.equal(masks[0], -2), tf.tile((jr[0] + (jr[1] if n_rings > 1 else zero)) / 2, [n_rings]), 0)
                # b += tf.where(tf.equal(self.masks2[i], -3), tf.tile(1j * jr_sqrt[i], [self.n_rings + 1]), 0) \
                #     + tf.where(tf.equal(self.masks2[i], 3), tf.tile(-1j * jr_sqrt[i], [self.n_rings + 1]), 0) # idler conj
                # TODO: contruct b for i == 0
            
            else:
                m += tf.where(tf.equal(self.masks[i], 1), tf.tile(1j * jr[i] / 2, [self.n_rings]), 0) \
                    + tf.where(tf.equal(self.masks[i], -1), tf.tile(-1j * jr[i] / 2, [self.n_rings]), 0) #\
                    # + tf.where(tf.equal(masks[i], 2), tf.tile((jr[i] + (jr[i + 1] if n_rings > (i + 1) else zero)) / 2, [n_rings]), 0) \
                    # + tf.where(tf.equal(masks[i], -2), tf.tile((jr[i] + (jr[i + 1] if n_rings > (i + 1) else zero)) / 2, [n_rings]), 0)
        
                # b += tf.where(tf.equal(self.masks2[i], 2), tf.tile(-1j * jr[i] / 2, [self.n_rings + 1]), 0) \
                #     + tf.where(tf.equal(self.masks2[i], -2), tf.tile(1j * jr[i] / 2, [self.n_rings + 1]), 0) \
                #     + tf.where(tf.equal(self.masks2[i], 3), tf.tile(-1j * jr[i] / 2, [self.n_rings + 1]), 0) \
                #     + tf.where(tf.equal(self.masks2[i], -3), tf.tile(1j * jr[i] / 2, [self.n_rings + 1]), 0)

                

            for j in tf.range(self.length):  
                m += tf.where(tf.equal(self.masks[i], j * 2 + 3), 1j * js[i][j] / 2, 0) \
                    + tf.where(tf.equal(self.masks[i], j * 2 + 4), 1j * tf.math.conj(js[i][j]) / 2, 0) \
                    + tf.where(tf.equal(self.masks[i], -j * 2 - 3), -1j * tf.math.conj(js[i][j])/ 2, 0) \
                    + tf.where(tf.equal(self.masks[i], -j * 2 - 4), -1j * js[i][j] / 2, 0)
        # print(m)
        # a = a + b
        # for i in tf.range(self.orth_itr):
        #     a = a + tf.matmul(b, a)
        #     b = tf.matmul(b, b)
            
        # u = tf.matmul(tf.linalg.inv(m + tf.cast(tf.eye(m.shape[-1]) * 0.001, dtype=tf.complex64)), a[self.nodes_t * 2 : self.nodes_t * 2 * (self.n_rings + 1)])
        u = tf.linalg.inv(m)
        # print(tf.linalg.eigvals(m).numpy())
        # print(tf.linalg.eigvals(u[0 : nodes_t * 2, 0: 2 * nodes_t]).numpy())
        u4 = tf.reverse(u[0 : self.nodes_t, self.nodes_t: 2 * self.nodes_t], [0]) * -1j * tf.math.sqrt(jr[0][self.nodes_t: 2 * self.nodes_t])
        u1 = tf.math.conj(u4)
        
        u = u * tf.reshape(tf.math.sqrt(y0s), [-1])
        
        u4_p = tf.reverse(tf.concat([u[0 : self.nodes_t, (i * 2 + 1) * self.nodes_t : (i * 2 + 2) * self.nodes_t] for i in range(self.n_rings + 1)], axis=1), [0])
        u2_p = tf.concat([u[self.nodes_t : 2 * self.nodes_t, (i * 2 + 1) * self.nodes_t : (i * 2 + 2) * self.nodes_t] for i in range(self.n_rings + 1)], axis=1)
        u3_p = tf.math.conj(u2_p)
        u1_p = tf.math.conj(u4_p)
        
        u1u2_p = tf.linalg.matmul(u1_p, tf.transpose(u2_p))
        u3u4_p = tf.linalg.matmul(u4_p, tf.transpose(u3_p))
        
        return (tf.reshape(tf.reverse(jr[0][0 : self.nodes_t], [0]), (self.nodes_t, 1)) * (u1 - 1j * tf.math.sqrt(jr[0][self.nodes_t : self.nodes_t * 2]) * u1u2_p) * (u4 + 1j * tf.math.sqrt(jr[0][self.nodes_t : self.nodes_t * 2]) * u3u4_p) * tf.constant(4, dtype=tf.complex64) * self.pi * self.pi)[self.padding : self.nodes + self.padding, self.padding : self.nodes + self.padding]

@tf.keras.saving.register_keras_serializable()
class JsiError(losses.Loss):
    def __init__(self, kernel=JsiKernel(28, 0, 5, 5, 5), reduction=losses.Reduction.AUTO, name='jsi_error'):
        super().__init__(reduction=reduction, name=name)
        self.kernel = kernel

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'kernel': self.kernel}
        
    def _error_calc(self, params):
        true, pred = params
        y_pred = self.kernel.call_flat(pred)
        y_pred = tf.linalg.l2_normalize(y_pred)
        has_nan = tf.math.reduce_any(tf.math.is_nan(tf.cast(y_pred, dtype=tf.float32)))
        if has_nan:
            return tf.constant(1.0, dtype=tf.float32)
        y_true = tf.cast(true[:, :, 0], tf.complex64)
        output = tf.cast(
            1 - tf.abs( tf.tensordot(tf.math.conj(y_true), y_pred, axes=2) * tf.tensordot(tf.math.conj(y_pred), y_true, axes=2) )
                       / (tf.abs((tf.tensordot(tf.math.conj(y_pred),  y_pred, axes=2)) * tf.tensordot(tf.math.conj(y_true), y_true, axes=2)))
            , tf.float32)
        return tf.constant(1.0, dtype=tf.float32) if tf.math.is_nan(output) else output
    def call(self, true, pred):
        error = tf.map_fn(self._error_calc, (true, pred), dtype=tf.float32)
        return tf.reduce_mean(error)

# returns heatmap figure of magnitude and phase
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

# def maskr(nodes, n_ring, n):
#     output = np.zeros([nodes * 2 * n_ring, nodes * 2 * n_ring], dtype = np.uint32)
#     pad = n * 2 * nodes
#     for i in range(nodes):
        
#         if n == 0:
#             output[i][nodes * 2 - 1 - i] = 1
#             output[nodes + i][nodes - i - 1] = -1
#         else:
#             output[pad - nodes * 2 + i][pad + i] = -1
#             output[pad - nodes + i][pad + nodes + i] = 1
#             # hermitian
#             output[pad + i][pad - nodes * 2 + i] = -1
#             output[pad + nodes + i][pad - nodes + i] = 1
#             # non-hermitian?
#             # output[pad + i][pad - nodes * 2 + i] = 1
#             # output[pad + nodes + i][pad - nodes + i] = -1

#         output[pad + i][pad + i] = 2
#         output[pad + nodes + i][pad + nodes + i] = -2
        
#         for j in [x + 1 for x in range(nodes - 1)]:
#             if i + j < nodes:
#                 output[pad + i][pad + i + j] = 2 * j + 1
#                 output[pad + nodes + i][pad + nodes + i + j] = -2 * j - 1
#             if i - j >= 0:
#                 output[pad + i][pad + i - j] = 2 * j + 2
#                 output[pad + nodes + i][pad + nodes + i - j] = -2 * j - 2
#     return output

# def mask2(nodes, n_ring, n):
#     output = np.zeros([nodes * 2 * (n_ring + 1), nodes * 2 * (n_ring + 1)], dtype = np.uint32)
#     pad = 0 if n == n_ring else n + 1 * 2 * nodes
#     for i in range(nodes):
#         output[pad + i][pad + i] = -1
#         output[pad + nodes + i][pad + nodes + i] = 1
#         if pad > 0:
#             output[pad - nodes * 2 + i][pad + i] = -2
#             output[pad - nodes + i][pad + nodes + i] = 2
#             output[pad + i][pad - nodes * 2 + i] = -3
#             output[pad + nodes + i][pad - nodes + i] = 3
#     return output
            
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


def jsi_backprop(init, EPOCHS=None, lr=None, train=True):
    if train and not EPOCHS:
        raise ValueError("training parameter \'EPOCHS\' not provided")
    
    nodes = init.get('nodes')
    padding = init.get('padding', 0)
    target = init.get('target', None)
    n_rings = init.get('n_rings', 1)
    orth_itr = init.get('orth_itr', 3)
    data = init.copy()
    nodes_t = nodes + 2 * padding
    length = init.get('length', nodes_t - 1)
    losses = np.zeros(int(EPOCHS if EPOCHS else 1), dtype=np.float32)
    kernel = JsiKernel(nodes, padding, n_rings, length, orth_itr)
    
    if n_rings < 1:
        raise ValueError("ring count cannot be less than 1, but got " + str(n_rings))

    @tf.function
    def pos_real_constraint(x):
        return tf.cast(tf.abs(tf.math.real(x)), tf.complex64)

    js = tf.Variable(init.get('js', np.ones((n_rings, length), dtype=np.complex64) / 10), name="coupling", dtype=tf.complex64)
    g = tf.Variable(init.get('g', 0.03), dtype=tf.complex64, name="g", constraint=pos_real_constraint)
    jr = tf.Variable(init.get('jr', np.ones((n_rings, nodes_t * 2), dtype=np.complex64) / 10), name="interring", dtype=tf.complex64, constraint=pos_real_constraint)
    y0s = tf.Variable(init.get('y0s', np.ones((n_rings, nodes_t * 2), dtype=np.complex64) / 10), name="loss", dtype=tf.complex64, constraint=pos_real_constraint)
    target_t = tf.zeros(shape=(nodes, nodes), dtype=tf.complex64)
    min_loss = 1.0
    best_params = [tf.Variable(p) for p in [js, jr, g, y0s]]
    
    if train:
        target_t = tf.constant(target, shape=(nodes, nodes), dtype=tf.complex64)
    output = None
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr if lr else 
                                         tf.keras.optimizers.schedules.ExponentialDecay(
                                             1e-3, decay_steps=100, decay_rate=0.9, staircase=True))

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

    @tf.function
    def train_step(loss_func, model, js, jr, g, y0s):
        with tf.GradientTape() as tape:
            pred = model(js, jr, g, y0s)
            loss = loss_func(pred)
            if tf.math.reduce_any(tf.math.is_nan(tf.cast(pred, dtype=tf.float32))):
                return tf.constant(-1, dtype=tf.float32)
            else:
                gradients = tape.gradient(loss, [js, jr, g, y0s])  # Compute gradients
                clipped = [tf.cast(tf.clip_by_value(tf.abs(grad), -0.1, 0.1), dtype=tf.complex64) * tf.exp(1j * tf.cast(tf.math.angle(grad), dtype=tf.complex64)) for grad in gradients]
                optimizer.apply_gradients(zip(clipped, [js, jr, g, y0s]))  # Update weights
                return loss
        
    try:
        tf.profiler.experimental.stop()
    except:
        pass
    try:
        # tf.profiler.experimental.start('log/' + datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S') + '.log')
        if train:
            # Create a checkpoint object
            ckpt = tf.train.Checkpoint(optimizer=optimizer, params=[js, jr, g, y0s])
            for j in trange(int(EPOCHS), desc="iterations"):
                loss = train_step(loss_func, kernel, js, jr, g, y0s)
                if loss.numpy() == -1: 
                    warnings.warn("Interrupted due to encountering NaN in the resulting JSI\n")
                    break
                else: 
                    losses[j] = loss.numpy()
                    if loss < min_loss:
                        min_loss = loss
                        best_params = [old.assign(new) for old, new in zip(best_params, [js, jr, g, y0s])]
                if j % 100 == 0:
                    ckpt.save('./chkpt/chkpt')
        else:
            losses = [init.get('loss', 0), ]

        # tf.profiler.experimental.stop()
    except KeyboardInterrupt:
        # tf.profiler.experimental.stop()
        print("Interrupted. Progress is saved in chkpt/")
        try:
            data['int'] = True
            return (data, losses, kernel(js, jr, g, y0s).numpy())
        except:
            try:
                sys.exit(130)
            except SystemExit:
                os._exit(130)

    data['js'] = js.numpy()
    data['jr'] = jr.numpy()
    data['g'] = g.numpy()
    data['y0s'] = y0s.numpy()
    data['loss'] = min_loss
    return (data, losses, kernel(best_params[0], best_params[1], best_params[2], best_params[3]).numpy())

def jsi_conv(param, filename, epochs=5):
    nodes = param.get('nodes')
    padding = param.get('padding', 0)
    n_rings = param.get('n_rings', 1)
    orth_itr = param.get('orth_itr', 3)
    nodes_t = nodes + 2 * padding
    length = param.get('length', nodes_t - 1)
    num_param = n_rings * length * 2 + nodes * 2 * n_rings * 2 + 1

    kernel = JsiKernel(nodes, padding, n_rings, length, orth_itr)

    # @tf.keras.saving.register_keras_serializable()
    # class JsiError(losses.Loss):
    #     def _error_calc(self, params):
    #         true, pred = params
    #         # reorganize parameters
    #         js = tf.cast(tf.reshape(pred[0:n_rings * length], (n_rings, length)), dtype=tf.complex64) * tf.exp(1j * tf.cast(tf.reshape(pred[n_rings * length:n_rings * length * 2], (n_rings, length)), dtype=tf.complex64))
    #         jr = tf.cast(tf.reshape(pred[n_rings*length*2 : n_rings*length*2+nodes_t*2*n_rings], (n_rings, nodes_t*2)), dtype=tf.complex64)
    #         g = tf.cast(pred[-1], dtype=tf.complex64)
    #         y0s = tf.cast(tf.reshape(pred[n_rings*length*2+nodes_t*2*n_rings:n_rings*length*2+nodes_t*4*n_rings], (n_rings, nodes_t*2)), dtype=tf.complex64)
            
    #         y_pred = kernel(js, jr, g, y0s)
    #         y_pred = tf.linalg.l2_normalize(y_pred)
    #         has_nan = tf.math.reduce_any(tf.math.is_nan(tf.cast(y_pred, dtype=tf.float32)))
    #         if has_nan:
    #             return tf.constant(1.0, dtype=tf.float32)
    #         y_true = tf.cast(true[:, :, 0], tf.complex64)
    #         output = tf.cast(
    #             1 - tf.abs( tf.tensordot(tf.math.conj(y_true), y_pred, axes=2) * tf.tensordot(tf.math.conj(y_pred), y_true, axes=2) )
    #                        / (tf.abs((tf.tensordot(tf.math.conj(y_pred),  y_pred, axes=2)) * tf.tensordot(tf.math.conj(y_true), y_true, axes=2)))
    #             , tf.float32)
    #         return tf.constant(1.0, dtype=tf.float32) if tf.math.is_nan(output) else output
    #     def call(self, true, pred):
    #         error = tf.map_fn(self._error_calc, (true, pred), dtype=tf.float32)

    #         return tf.reduce_mean(error)
    
    def create_cnn_model():
        model = models.Sequential([
            layers.Conv2D(32, (11, 11), padding='same', activation='relu', input_shape=(28, 28, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
            layers.Flatten(),
            layers.Dense(4096, activation='softmax'),
            layers.Dense(4096, activation='relu'),
            layers.Dense(2048, activation='relu'),
            layers.Dense(1024, activation='relu'),
            layers.Dense(num_param, activation='relu'),
            layers.Dense(num_param)
        ])
        return model
        
    def _parse_function(proto):
        # Define features
        features = {
            'data': tf.io.FixedLenFeature([nodes * nodes], tf.float32),
            'label': tf.io.VarLenFeature(tf.float32)
        }
        # Load one example
        parsed = tf.io.parse_single_example(proto, features)
        
        # Extract the image as a 28x28 array
        parsed['data'] = tf.reshape(parsed['data'], (nodes, nodes, 1))
        
        return parsed['data'], parsed['data']

    dataset = None
    if filename == 'mnist':
        (train_images, _), (_, _) = mnist.load_data()
        train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
        dataset = tf.data.Dataset.from_tensor_slices((train_images, train_images))
    else:
        # Create a dataset from the TFRecord file
        dataset = tf.data.TFRecordDataset(filenames=filename)
    
        dataset = dataset.map(_parse_function)

    # Shuffle, batch, and prefetch the dataset
    dataset = dataset.shuffle(1024).batch(16).prefetch(tf.data.experimental.AUTOTUNE)

    optimizer = tf.keras.optimizers.Adam(1e-4)
    
    model = param.get('model')
    if not 'model' in param or model is None:
        model = create_cnn_model()
        model.compile(optimizer=optimizer,
                  loss=JsiError(kernel))

    if isinstance(model, str):
        model = models.load_model(model)
        
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='./nn_chkpt',
        monitor='loss',
        mode='min',
        save_best_only=True
    )

    model.fit(dataset, epochs=epochs, callbacks=[model_checkpoint_callback])
    return model