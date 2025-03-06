import argparse 

if __name__ == "__main__":
    args= argparse.ArgumentParser()
    args.add_argument("--name", "-n", default="akshay", type=str)
    args.add_argument("--age", "-a", default="25.0", type=float)
    parsed_args= args.parse_args()
    print(parsed_args.name, parsed_args.age) 