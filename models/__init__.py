from .cnn4conv import CNN4Conv
from .resnet import *
    
    
def get_model(args):
    if args.model == 'cnn4conv':
        net_glob = CNN4Conv(in_channels=args.in_channels, num_classes=args.num_classes, args=args).to(args.device)
    elif args.model == 'resnet18':
        # NOTE: here we use pytorch model
        net_glob = resnet18(num_classes=args.num_classes, pretrained=args.pretrained).to(args.device)
    elif args.model == 'resnet50':
        net_glob = resnet50(num_classes=args.num_classes, pretrained=args.pretrained).to(args.device)
    else:
        exit('Error: unrecognized model')

    return net_glob