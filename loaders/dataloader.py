from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

class FolderLoader :
    def __init__(self, args, cuda):
        print("Loading Data...")
        if args.name == 'ImageFolder':
            self.folderloader(args, cuda)
        elif args.name == 'MNIST':
            self.mnistloader(args, cuda)
        else:
            raise ValueError('The dataset name should be specified.')
    def folderloader(self, args, cuda):
        # Data Loading
        root_dir = os.path.join(os.path.dirname(__file__), '../', args.root)
        kwargs = {'num_workers': args.dataloader_workers, 'pin_memory': args.pin_memory} if cuda else {}
        transform_train = transforms.Compose([
            transforms.Scale(args.image_size + 30),
            transforms.CenterCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([ 0.5, 0.5, 0.5 ], [ 0.22, 0.22, 0.22 ])
        ])

        transform_test = transforms.Compose([
            transforms.Scale(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize([ 0.5, 0.5, 0.5 ], [ 0.22, 0.22, 0.22 ])
        ])

        train_root = root_dir + '/train'
        train_set = datasets.ImageFolder(root=train_root, transform=transform_train)
        self.train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=args.shuffle,
                                       **kwargs)

        test_root = root_dir + '/test'
        test_set = datasets.ImageFolder(root=test_root, transform=transform_test)
        self.test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        print("Data loaded successfully\n")

    def mnistloader(self):
        pass
