from app.Problem.Problem import Problem

# TEST

if __name__ == "__main__":
    M = [[30, 29, 19, 50],
     [39, 69, 6, 73],
     [3, 86, 58, 60],
     [72, 74, 51, 15],
     [24, 47, 22, 90]]
    
    p = Problem(4,5,M)

    print("--- Display test ---")
    print(p)

    print("--- Test on known sequence ---")
    result = p.evaluate([0, 2, 3, 1])
    if result == 1170:
        print(f'Unit test successful, for the sequence [1, 3, 4, 2], we get {result}')
    else:
        print(f"Unit test failed, for the sequence [1, 3, 4, 2], we get {result}'")