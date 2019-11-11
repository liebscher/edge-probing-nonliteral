import sys

def check_args(args):
    accepted = [
        "--task",
        "--log"
    ]

    for arg in args:
        assert arg in accepted, f"Unaccepted argument given: \"{arg}\""

def main():
    args = len(sys.argv)

    params = {sys.argv[i]: sys.argv[i + 1] for i in range(1, args, 2)}

    check_args(params.keys())

    model_setup = dict()

    for param in params:
        value = params[param]
        # print(f"Set {param} as {value}")

        if param == "--task":
            model_setup["model_task"] = value.split(",")
        elif param == "--log":
            pass


    print(model_setup)

    exit(1)

if __name__ == "__main__":
    main()
    exit(0)