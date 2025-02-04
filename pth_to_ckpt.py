import torch
import argparse

def pth_to_pth_tar(input_pth, output_pth_tar, epoch=100, optimizer_state=None):
    state_dict = torch.load(input_pth, map_location='cpu')
    checkpoint = {
        'epoch': epoch,
        'state_dict': state_dict,
        'optimizer': optimizer_state if optimizer_state is not None else {}}
    torch.save(checkpoint, output_pth_tar)
    print(f"{input_pth} to {output_pth_tar} with additional fields")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=".pth to .pth.tar with checkpoint fields.")
    parser.add_argument("input_pth", type=str, help="path to input .pth file")
    parser.add_argument("output_pth_tar", type=str, help="path to output .pth.tar file")
    parser.add_argument("--epoch", type=int, default=100, help="epoch number to include in checkpoint")
    parser.add_argument("--optimizer", type=str, default=None, help="path to optional optimizer state .pth file")
    
    args = parser.parse_args()
    
    optimizer_state = None
    if args.optimizer:
        optimizer_state = torch.load(args.optimizer, map_location='cpu')
    
    pth_to_pth_tar(args.input_pth, args.output_pth_tar, args.epoch, optimizer_state)
