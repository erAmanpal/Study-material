def fun(n):
    if n>2:
        fun(n-1)
        fun(n-2)
        fun(n-3)
    print(str(n)+" ",end="")

fun(5)
