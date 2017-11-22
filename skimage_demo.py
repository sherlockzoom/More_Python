# coding=utf-8
"""
@created: 17-11-22
@author: zyl
"""


from skimage import data, io, filters

image = data.coins()
# ... or any other NumPy array!
edges = filters.sobel(image)
io.imshow(edges)
io.show()