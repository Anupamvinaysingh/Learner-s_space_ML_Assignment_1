import numpy as np
array_1d = np.random.uniform(0, 10, 20)
print("Array:", array_1d)
rounded_array = np.round(array_1d, 2)
print("Rounded array:", rounded_array)
print("Min:", np.min(array_1d))
print("Max:", np.max(array_1d))
print("Median:", np.median(array_1d))


modified_array = np.where(array_1d < 5, np.square(array_1d), array_1d)
print("Modified array:", np.round(modified_array, 2))


def numpy_alternate_sort(array):
    sorted_array = np.sort(array)
    result = []
    i, j = 0, len(sorted_array) - 1
    while i <= j:
        result.append(sorted_array[i])
        if i != j:
            result.append(sorted_array[j])
        i += 1
        j -= 1
    return np.array(result)

alternate_sorted = numpy_alternate_sort(array_1d)
print("Alternate sorted array:", np.round(alternate_sorted, 2))