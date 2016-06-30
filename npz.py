import numpy

from chainer import cuda
from chainer import serializer

class NpzDeserializer(serializer.Deserializer):

    """Deserializer for NPZ format.

    This is the standard deserializer in Chainer. This deserializer can be used
    to read an object serialized by :func:`save_npz`.

    Args:
        npz: `npz` file object.
        path: The base path that the deserialization starts from.

    """
    def __init__(self, npz, path=''):
        self.npz = npz
        self.path = path

    def __getitem__(self, key):
        key = key.strip('/')
        return NpzDeserializer(self.npz, self.path + key + '/')

    def __call__(self, key, value):
        key = key.lstrip('/')
        dataset = self.npz[self.path + key]
        if isinstance(value, numpy.ndarray):
	    dataset = dataset.view("<f4")
	    dataset = dataset.reshape(value.shape)
            numpy.copyto(value, dataset)
        elif isinstance(value, cuda.ndarray):
            value.set(numpy.asarray(dataset))
        else:
            value = type(value)(numpy.asarray(dataset))
        return value


def load_npz(filename, obj):
    """Loads an object from the file in NPZ format.

    This is a short-cut function to load from an `.npz` file that contains only
    one object.

    Args:
        filename (str): Name of the file to be loaded.
        obj: Object to be deserialized. It must support serialization protocol.

    """
    with numpy.load(filename) as f:
	d = NpzDeserializer(f)
	d.load(obj)
