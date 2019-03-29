import argparse

def int2str(val: int):
  val = str(val)
  return " ".join(list(val))

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="create a sample dataset")

  parser.add_argument("--max", type=int, default=100)
  parser.add_argument("--type", type=str, default="+", choices=["+","-"])
  parser.add_argument("--path", type=str, required=True)

  args = parser.parse_args()

  prob, ans = [], []
  for i in range(args.max):
    for j in range(args.max):
      prob.append("{} {} {}".format(int2str(i), args.type, int2str(j)))
      ans.append(int2str(i+j))

  with open("{}/problem".format(args.path), "w", newline="\n", encoding="utf-8") as f:
    f.write("\n".join(prob))
  with open("{}/answer".format(args.path), "w", newline="\n", encoding="utf-8") as f:
    f.write("\n".join(ans))
