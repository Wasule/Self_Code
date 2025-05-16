# -*- coding: utf-8 -*-
"""
Created on Tue May 13 12:31:50 2025

@author: LENOVO
"""
def longest_consecutive_sequence(nums):
    if not nums:
        return 0

    num_set = set(nums)
    longest = 0

    for num in num_set:
        # Only start counting if `num` is the beginning of a sequence
        if num - 1 not in num_set:
            current = num
            streak = 1

            while current + 1 in num_set:
                current += 1
                streak += 1

            longest = max(longest, streak)

    return longest

# Example test
arr = [100, 4, 200, 1, 3, 2]
print("Longest consecutive sequence length is:", longest_consecutive_sequence(arr))
