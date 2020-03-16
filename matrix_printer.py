import numpy as np

mat_map = np.zeros(shape=(100, 100))

# boundary
mat_map[0, :] = -1
mat_map[99, :] = -1
mat_map[:, 0] = -1
mat_map[:, 99] = -1

# obstacle
# map 1
mat_map[30:70, 30:70] = -1

# map 2
# mat_map[0:40, 70:80] = -1
# mat_map[0:40, 30:40] = -1
# mat_map[50:100, 50:60] = -1

#points
start_point = (10, 10)
current_point = tuple(start_point)
end_point = (90, 90)
mat_map[current_point] = 1
mat_map[end_point] = 2

# fp = open("map.txt", "w")

for i in range(len(mat_map)):
#     print>>fp,str(mat_map[i])
    printf(str(mat_map[i]))


print(len(mat_map))
 #   print >> fp, "\n"
 #   print("\n")

