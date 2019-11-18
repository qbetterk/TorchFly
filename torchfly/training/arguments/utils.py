import argparse

def args_dict_to_list(args):
    res = " ".join([ "--" + k + " " + str(v) for k, v in args.items()])
    res = res.split()
    res = [item for item in res if item != 'None']
    return res