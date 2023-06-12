import numpy as np

# Create an empty array
empty_array = np.array([])

# Save the empty array to a file
np.save("final_test_acc.npy", empty_array)
np.save("worst_test_acc.npy", empty_array)

print(empty_