data = eval(input()) # unsafe
n = 8

for i in range(n+1):
    if i == 0:
        print("  ", end="")
        for j in range(n):
            print(j, end = " ")
        print()
        for j in range(n):
            print("_", end = " ")
    else:
        print("\n" + str(i-1), end= "|")
        for j in range(n):
            if (i-1,j) in data:
                print(data.get((i-1,j)), end = " ")
            else:
                print("  ", end="")

