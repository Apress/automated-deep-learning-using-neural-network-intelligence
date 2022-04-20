import tensorflow_datasets as tfds

ds, info = tfds.load('mnist', split = 'train', with_info = True)
fig = tfds.show_examples(ds, info)
fig.show()
