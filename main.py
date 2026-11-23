import argparse
from scripts.train_stage1 import main as train_stage1
from scripts.train_stage2 import main as train_stage2
from scripts.inference import main as inference


def main():
    parser = argparse.ArgumentParser(description="TcALign with Diffusion")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # 只定义子命令路由，具体参数由各脚本自行解析
    subparsers.add_parser("train_stage1", help="Train stage 1: Text-Conditional 3D Editor")
    subparsers.add_parser("train_stage2", help="Train stage 2: Conditional Diffusion Mapping")
    subparsers.add_parser("inference", help="Run inference")

    subparsers.add_parser("evaluate", help="Evaluate a checkpoint on paired portraits", add_help=False)

    # parse_known_args: 子命令后的参数放入 remaining，传给内层解析
    args, remaining = parser.parse_known_args()

    if args.command == "train_stage1":
        train_stage1(remaining)
    elif args.command == "train_stage2":
        train_stage2(remaining)
    elif args.command == "inference":
        inference(remaining)
    elif args.command == "evaluate":
        from scripts.evaluate import main as evaluate

        evaluate(remaining)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
