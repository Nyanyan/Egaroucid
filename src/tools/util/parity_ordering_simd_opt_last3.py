from itertools import permutations

places_lst = list(permutations([0, 0, 0, 4, 4, 4, 32, 32, 32, 36, 36, 36], 3))

parity_ordering_shuffle_mask = [-1 for _ in range(64)]

for places in places_lst:
    p0, p1, p2 = places
    parities = (p0 ^ p1) | ((p1 ^ p2) >> 1)
    place_count = []
    for elem in [0, 4, 32, 36]:
        place_count.append(places.count(elem))
    sorted_place_count = sorted(place_count)
    need_to_sort = (sorted_place_count == [0, 0, 1, 2])
    sort_lst = [0, 1, 2]
    if need_to_sort: # 2-1-0
        sort_lst = []
        no_priority = []
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                if places[i] == places[j]:
                    no_priority.append(i)
                    break
            else:
                sort_lst.append(i)
        sort_lst.extend(no_priority)
        #print(places, sort_lst)
    sort_lst = list(reversed(sort_lst))
    sort_32bit = 0
    for i in range(3):
        sort_32bit <<= 8
        sort_32bit |= sort_lst[i]
    parity_ordering_shuffle_mask[parities] = sort_32bit

for i in range(64):
    if parity_ordering_shuffle_mask[i] == -1:
        print('0x00000', end='')
    else:
        print('0x' + hex(parity_ordering_shuffle_mask[i]).upper()[2:], end='')
    print('U, ', end='')
    if i % 8 == 7:
        print('')
