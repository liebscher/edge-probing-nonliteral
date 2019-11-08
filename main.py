import sys

def main():
    args = len(sys.argv)

    params = {sys.argv[i]: sys.argv[i + 1] for i in range(1, args, 2)}

    for param in params:
        value = params[param]
        print(f"Set {param} as {value}")

if __name__ == "__main__":
    main()
    exit(0)