from typing import Union, List
def sorter(arr: Union[List[int],List[float]]) -> Union[List[int],List[float]]:
    for i in range(len(arr)):
        for j in range(len(arr) - 1):
            if arr[j] > arr[j + 1]:
                temp = arr[j]
                arr[j] = arr[j + 1]
                arr[j + 1] = temp
    return arr