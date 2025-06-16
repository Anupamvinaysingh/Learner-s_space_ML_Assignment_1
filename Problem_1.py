import numpy as np
array_2d = np.random.randint(1, 51, size=(5, 4))
print("Array:\n", array_2d)
anti_diagonal = [array_2d[i, -(i + 1)] for i in range(min(array_2d.shape))]
print("Anti-diagonal elements:", anti_diagonal)
row_max = np.max(array_2d, axis=1)
print("Max value in each row:", row_max)
mean_val = np.mean(array_2d)
less_equal_mean = array_2d[array_2d <= mean_val]              #array containing only the elements less than or equal to the overall mean of the array.
print("Elements <= mean (%.2f):" % mean_val, less_equal_mean) #printed this array also, not asked in the question
def numpy_boundary_traversal(matrix):
    top = [int(x) for x in matrix[0,:]]
    right = [int(x) for x in matrix[1:-1, -1]]
    bottom = [int(x) for x in matrix[-1,::-1]]
    left = [int(x) for x in matrix[-2:0:-1, 0]]
    return top + right + bottom + left                        #function that takes a NumPy matrix and returns a list of elements visited along the boundary of the array in clockwise order, starting from the top-left corner.

boundary_elements = numpy_boundary_traversal(array_2d)
print("Boundary traversal:", boundary_elements)
