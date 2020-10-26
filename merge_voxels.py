import struct
import numpy as np
import scipy as sp
import sklearn.cluster
import sys

def as_uint(data):
    return struct.unpack("I", data)[0]

def as_float(data):
    return struct.unpack("f", data)[0]

path = "./"

def read_matrix(fin):
    matrix_size = as_uint(fin.read(4))
    matrix = np.zeros((matrix_size, matrix_size))
    for i in range(matrix_size):
        for j in range(matrix_size):
            matrix[i, j] = as_float(fin.read(4))
    return matrix
    
def write_clustering(clustering, fout):
    for cluster in clustering:
        fout.write(struct.pack("I", len(cluster)))
        for value in cluster:
            fout.write(struct.pack("I", value))
    
def read_colors(length):
    filename = path + "Colors" + str(length)
    with open(filename, "rb") as fin:
        colors_count = as_uint(fin.read(4))
        if colors_count != length:
            print("Wrong colors count!")
        colors = np.zeros((colors_count, 3))
        for i in range(colors_count):
            for j in range(3):
                colors[i][j] = as_float(fin.read(4))
    return colors

def read_voxels(length):
    filename = path + "VoxelIds" + str(length)
    with open(filename, "rb") as fin:
        voxels_count = as_uint(fin.read(4))
        if voxels_count != length:
            print("Wrong voxels count!")
        voxels = np.zeros(voxels_count, "int")
        for i in range(voxels_count):
            voxels[i] = as_uint(fin.read(4))
    return voxels

def compute_lighting_matrix(ff, colors):
    col = colors[:, None]
    # return col * ff
    return sp.linalg.inv(np.eye(*ff.shape) - col * ff) - np.eye(*ff.shape)

filename = path + "tmp_ff_dist.bin"
filenameOut = path + "tmp_clust_data.bin"
with open(filename, "rb") as fin, open(filenameOut, "wb") as fout:
    voxelsCount = as_uint(fin.read(4))
    for voxel in range(voxelsCount):
        ffDist = read_matrix(fin)
        if ffDist.shape[0] == 0:
            continue
        clust = sklearn.cluster.AgglomerativeClustering(3, affinity="precomputed", linkage="average")
        clust.fit(ffDist)
        clustering = [[], [], []]
        for i, lab in enumerate(clust.labels_):
            clustering[lab].append(i)

        write_clustering(clustering, fout)
