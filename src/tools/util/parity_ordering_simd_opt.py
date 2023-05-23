from itertools import permutations

places_lst = list(permutations([0, 0, 0, 0, 4, 4, 4, 4, 32, 32, 32, 32, 36, 36, 36, 36], 4))

parities_lst = [[] for _ in range(64)]

parity_ordering_shuffle_mask = [-1 for _ in range(64)]

for places in places_lst:
    p0, p1, p2, p3 = places
    parities = (p0 ^ p1) | ((p1 ^ p2) >> 1) | ((p2 ^ p3) >> 2)
    parities_lst[parities].append([p0, p1, p2, p3])
    place_count = []
    for elem in [0, 4, 32, 36]:
        place_count.append(places.count(elem))
    sorted_place_count = sorted(place_count)
    need_to_sort = (sorted_place_count == [0, 1, 1, 2]) and (place_count != sorted_place_count)
    if need_to_sort:
        print(parities, places)
