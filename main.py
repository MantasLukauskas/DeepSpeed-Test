# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

# myls.py
# Import the argparse library
import argparse
import timeit
from transformers import pipeline
import time
import os
import sys
import pandas as pd


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='List the content of a folder')

    # Add the arguments
    # parser.add_argument('--number',
    #                        metavar='num',
    #                        type=int,
    #                        default = 2,
    #                        help='the path to list')

    parser.add_argument('--model_dir', type=str, help='Input dir for videos')
    parser.add_argument('--output_dir', type=str, help='Input dir for videos')
    parser.add_argument('--input_text', type=str, help='Input dir for videos')
    parser.add_argument('--file_name', default="results.txt", type=str, help='Input dir for videos')
    parser.add_argument('--device', type=int, help='Input dir for videos')

    df = pd.DataFrame(columns=['Model', "Device", 'No. Input', 'Time'])

    args = parser.parse_args()

    for model in ["distilgpt2", "gpt2", "gpt2-medium","gpt2-large", "gpt2-xl", "EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B"]:
        for device in [-1, 0]:

            try:
                # generator = pipeline("text-generation", model=args.model_dir, device=args.device)
                generator = pipeline("text-generation", model=model, device=device)
                # generator = pipeline("text-generation", model='EleutherAI/gpt-neo-125M', device=0)

                # list = []
                with open('input.txt') as f:
                    for line in f:
                        t0 = time.time()
                        generator(line, do_sample=True, max_length=100, num_return_sequences=10)
                        total = time.time() - t0
                        print(total)
                        # list.append(total)
                        # append rows to an empty DataFrame
                        df = df.append({'Time': total,
                                        "Device": device,
                                        'No. Input': len(line.strip().split()),
                                        "Model": model
                                        },
                                       ignore_index=True)

                # Execute the parse_args() method
                #
                # textfile = open(args.file_name, "w")
                # for element in list:
                #     textfile.write(str(element) + "\n")
                # textfile.close()`
            except:
                print("CUDA too small. Skipping this step")

    df.to_csv("Results.csv")
    # number = args.number
    #
    # list = []
    # for i in range (number):
    #     list.append(i)
    #
    # textfile = open(args.file_name, "w")
    # for element in list:
    #     textfile.write(str(element) + "\n")
    # textfile.close()

    # print(f"Your number is : {list}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
