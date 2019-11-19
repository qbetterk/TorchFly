import argparse

def args_dict_to_list(args):
    res = "\t".join([ "--" + k + "\t" + str(v) for k, v in args.items()])
    res = res.split("\t")
    res = [item for item in res if item != 'None']
    return res