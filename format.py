data = eval(input()) # unsafe
n = 8

for i in range(n):
    print()
    for j in range(n):
        if (i,j) in data:
            print(data.get((i,j)), end = " ")
        else:
            print("  ", end="")

