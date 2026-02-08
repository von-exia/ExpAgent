def add(a, b):
    return a+b

def multiply(a, b):
 return a * b

# Test sample
if __name__ == "__main__":
 print("Testing multiply function:")
 print(f"2 * 3 = {multiply(2, 3)}")
 print(f"-1 * 5 = {multiply(-1, 5)}")
 print(f"0 * 10 = {multiply(0, 10)}")

def multiply_new(a, b):
 """Multiplies two numbers, handling integers and floats."""
 return a * b

# Expanded test sample
if __name__ == "__main__":
 print("Testing multiply function:")
 print(f"2 * 3 = {multiply(2, 3)}")
 print(f"-1 * 5 = {multiply(-1, 5)}")
 print(f"0 * 10 = {multiply(0, 10)}")
 print("\nTesting multiply_new function:")
 print(f"2.5 * 4 = {multiply_new(2.5, 4)}")
 print(f"-3 * 2.5 = {multiply_new(-3, 2.5)}")
 print(f"0 * 7.2 = {multiply_new(0, 7.2)}")