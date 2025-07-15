def sum_of_squares_of_digits(num):
    sum = 0
    while num > 0:
        digit = num % 10
        sum += digit * digit
        num = num // 10
    return sum

def is_happy_number(n):
    seen_numbers = set()
    
    while n != 1 and n not in seen_numbers:
        seen_numbers.add(n)
        n = sum_of_squares_of_digits(n)
    
    return n == 1

# Example usage
if __name__ == "__main__":
    number = int(input("Enter a number: "))
    if is_happy_number(number):
        print(f"{number} is a happy number")
    else:
        print(f"{number} is not a happy number")
