def main():
    n = int(input())
    arr = [0] * 1000001

    for i in range(2, 1000000):
        k = i
        k += i
        while k < 1000000:
            arr[k] = 1
            k += i

    for _ in range(n):
        a = int(input())
        d = a
        for i in range(2, 1000001):
            if arr[i] == 0:
                a -= 1
            if a == 0:
                print(d * i + d)
                break

if __name__ == "__main__":
    main()
