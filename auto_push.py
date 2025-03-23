import os
import argparse

def auto_push(commit_message='auto push'):
    os.system("git add .")
    os.system(f"git commit -m {commit_message}")
    os.system("git push -u origin main")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Auto-push script for Git.")
    parser.add_argument("-m","--message",required=True, help="commit message")

    args = parser.parse_args()
    auto_push(args.message)