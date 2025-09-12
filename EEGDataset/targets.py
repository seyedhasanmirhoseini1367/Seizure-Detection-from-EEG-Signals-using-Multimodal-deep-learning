
consensus_by_id = df.groupby(id)['expert_consensus'].agg(list)

from collections import Counter

def get_majority_consensus(consensus_list):  

    # Count the frequency of each label in the consensus list  
    counter = Counter(consensus_list)

    # Get the two most frequent labels and their counts
    most_common = counter.most_common(2)

    # Check if there's a clear majority
    # len(most_common) == 1 If only one label is the most frequent
    
    # most_common[0][1] > most_common[1][1]:  If the first label is strictly more frequent
    
    if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:        
        return most_common[0][0] 
    else:
        # If there's no clear majority        
        return None  


# Example 1: Clear majority
consensus_list = ['A', 'A', 'A', 'B', 'C']
result = get_majority_consensus(consensus_list)
# Let's break it down:
# 1. Counter(consensus_list) creates: {'A': 3, 'B': 1, 'C': 1}
# 2. most_common(2) returns: [('A', 3), ('B', 1)]
# 3. Since 3 > 1, 'A' is returned
print("Example 1:", result)  
# Output: 'A'

# Example 2: Tie between valuesconsensus_list = ['A', 'A', 'B', 'B', 'C']
result = get_majority_consensus(consensus_list)
# 1. Counter(consensus_list) creates: {'A': 2, 'B': 2, 'C': 1}
# 2. most_common(2) returns: [('A', 2), ('B', 2)]
# 3. Since both have count 2, None is returned
print("Example 2:", result)  # Output: None

# Example 3: Single valueconsensus_list = ['A', 'A', 'A']
result = get_majority_consensus(consensus_list)
# 1. Counter(consensus_list) creates: {'A': 3}
# 2. most_common(2) returns: [('A', 3)]
# 3. Since there's only one value, 'A' is returned
print("Example 3:", result) # Output: 'A'

consensus_list = ['GRDA', 'GRDA', 'GRDA', 'Seizure', 'Seizure', 'Seizure']
result = get_majority_consensus(consensus_list)
# 1. Counter(consensus_list) creates: {'GRDA': 3, 'Seizure': 3}
# 2. most_common(2) returns: [('GRDA', 3), ('Seizure', 3)]
# 3. Since both have count 2, None is returned
print("Example 3:", result) # Output: None

consensus_list = ['LPD', 'LPD', 'LPD', 'Seizure']
result = get_majority_consensus(consensus_list)
# 1. Counter(consensus_list) creates: {'LPD': 3, 'Seizure': 1}
# 2. most_common(2) returns: [('LPD', 3), ('Seizure', 1)]
# 3. Since both have count 2, LPD is returned
print("Example 5:", result) # Output: LPD
