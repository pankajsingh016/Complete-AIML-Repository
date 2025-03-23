import argparse
import subprocess

def auto_push(commit_message='auto push'):
    subprocess.run(["git","add","."],check=True)
    subprocess.run(["git","commit","-m",commit_message],check=True)
    subprocess.run(["git","push","-u","origin","main"],check=True)
    print("Successfully Pushed the code!")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Auto-push script for Git.")
    parser.add_argument("-m","--message",required=True, help="commit message")

    args = parser.parse_args()
    auto_push(args.message)