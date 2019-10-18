# python 3 program to count number of swaps required 
# to sort an array when only swapping of adjacent 
# elements is allowed. 
#include <bits/stdc++.h> 

#This function merges two sorted arrays and returns inversion count in the arrays.*/ 
def merge(arr, temp, left, mid, right): 
	inv_count = 0

	i = left #i is index for left subarray*/ 
	j = mid #i is index for right subarray*/ 
	k = left #i is index for resultant merged subarray*/ 
	while ((i <= mid - 1) and (j <= right)): 
		if (arr[i] <= arr[j]): 
			temp[k] = arr[i] 
			k += 1
			i += 1
		else: 
			temp[k] = arr[j] 
			k += 1
			j += 1

			#this is tricky -- see above explanation/ 
			# diagram for merge()*/ 
			inv_count = inv_count + (mid - i) 

	#Copy the remaining elements of left subarray 
	# (if there are any) to temp*/ 
	while (i <= mid - 1): 
		temp[k] = arr[i] 
		k += 1
		i += 1

	#Copy the remaining elements of right subarray 
	# (if there are any) to temp*/ 
	while (j <= right): 
		temp[k] = arr[j] 
		k += 1
		j += 1

	# Copy back the merged elements to original array*/ 
	for i in range(left,right+1,1): 
		arr[i] = temp[i] 

	return inv_count 

#An auxiliary recursive function that sorts the input 
# array and returns the number of inversions in the 
# array. */ 
def _mergeSort(arr, temp, left, right): 
	inv_count = 0
	if (right > left): 
		# Divide the array into two parts and call 
		#_mergeSortAndCountInv() 
		# for each of the parts */ 
		mid = int((right + left)/2) 

		#Inversion count will be sum of inversions in 
		# left-part, right-part and number of inversions 
		# in merging */ 
		inv_count = _mergeSort(arr, temp, left, mid) 
		inv_count += _mergeSort(arr, temp, mid+1, right) 

		# Merge the two parts*/ 
		inv_count += merge(arr, temp, left, mid+1, right) 

	return inv_count 

#This function sorts the input array and returns the 
#number of inversions in the array */ 
def countSwaps(arr, n): 
	temp = [0 for i in range(n)] 
	return _mergeSort(arr, temp, 0, n - 1) 

# Driver progra to test above functions */ 
if __name__ == '__main__': 
	arr = [1, 20, 6, 4, 5] 
	n = len(arr) 
	print("Number of swaps is",countSwaps(arr, n)) 
	

# This code is contributed by 
# Surendra_Gangwar 

