#%%
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 20, 100)  # Create a list of evenly-spaced numbers over the range
plt.plot(x, np.sin(x))       # Plot the sine of each x point
plt.show()                   # Display the plot


msg = "Hello World"
print(msg)

#%%
import tensorflow as tf
tf.test.is_gpu_available()

#%%
tf.test.is_built_with_cuda()

