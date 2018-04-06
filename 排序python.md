import numpy as np


def bouble_sort(array):
    for i in range(len(array)-1):
        for j in range(i+1,len(array)):
            if array[i]>array[j]:
                array[i],array[j] = array[j],array[i]
    return array
def choise_sort(array):
    for i in range(len(array)-1):
        x = i
        for j  in range(i+1,len(array)):
            if array[x]>array[j]:
                x = j
        array[x],array[i] = array[i],array[x]
    return array

def insert_sort(array):
    for i in range(len(array)):
        for j in range(i):
            if array[i]<=array[j]:
                array.insert(j,array.pop(i))
                break
    return array

def quick_sort(array):
    def sort(begain,end):
        if begain>end:
            return
        l,r = begain,end
        p = array[begain]
        while l<r:
            while l<r and p<array[r]:
                r -= 1
            while l<r and p>=array[l]:
                l += 1
            array[l],array[r] = array[r],array[l]
        array[l],array[begain] = array[begain],array[l]
        sort(begain,l-1)
        sort(l+1,end)
    sort(0,len(array)-1)
    return array

def shell_sort(array):
    gap = len(array)//2
    while gap > 0:
        for i in range(len(array)//gap):
            for j in range(i):
                if array[i*gap]<array[j*gap]:
                    array.insert(j*gap,array.pop(i*gap))
                    break
        gap = gap//2
    return array

def heap_sort(array):
    def adjust_heap(parent):
        child = 2*parent+1
        while child <len(heap):
            if child+1 <len(heap):
                if heap[child+1]>heap[child]:
                    child += 1
            if heap[parent]>=heap[child]:
                break
            heap[child],heap[parent] = heap[parent],heap[child]
            parent,child = child,2*child+1
   
    heap,array = array.copy(),[]
    for i in range(len(heap)//2-1,-1,-1):
        adjust_heap(i)
    for i in range(len(heap)):
        heap[0],heap[-1] = heap[-1],heap[0]
        array.insert(0,heap.pop())
        adjust_heap(0)
    return array            
def merge_sort(array):
    def merge(array_l,array_r):
        array = []
        while len(array_l) and len(array_r):
            if array_l[0]>array_r[0]:
                array.append(array_r.pop(0))
            else:
                array.append(array_l.pop(0))
        if array_l:
            array += array_l
        if array_r:
            array += array_r
        return array
    def recursive(array):
        if len(array)==1:
            return array
        mid = len(array)//2
        array_l = recursive(array[:mid])
        array_r = recursive(array[mid:])
        return merge(array_l,array_r)
    return recursive(array)
def radix_sort(array):
    buck,digit =[[]],0
    while len(buck[0]) != len(array):
        buck = [[],[],[],[],[],[],[],[],[],[]]
        for i  in range(len(array)):
            num = (array[i]//10**digit)%10
            buck[num].append(array[i])
        array = []
        for i  in range(len(buck)):
            array += buck[i]
        digit += 1
    return array
                

a = np.random.randint(0,300,(16))
b = radix_sort(list(a))
