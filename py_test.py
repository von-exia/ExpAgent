def add(a, b):
    return a+b

def multiply(a, b):
 return a * b

if __name__ == "__main__":
 # Test sample for multiply function
 result = multiply(5, 3)
 print(f"Multiplication test: 5 * 3 = {result}")
 assert result == 15, f"Expected 15, got {result}"
 print("Test passed!")