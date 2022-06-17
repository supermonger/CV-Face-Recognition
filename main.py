from train import train, val, test, preprocessing, fixed_seed
import argparse

if __name__ == "__main__":
    # fixed_seed(30678)
    # # checkpoint = "best.pt"
    # batch_size = 32
    # train_loader, valid_loader = preprocessing(batch_size)
    # train(train_loader, valid_loader)
    # # test(checkpoint, valid_loader, batch_size)
    parser = argparse.ArgumentParser(description='main code of Facial landmark detection')
    parser.add_argument('-m','--mode', choices=['train', 'test'], required=True, help='choose mode')
    parser.add_argument('-c','--checkpoint', default='best.pt', help='checkpoint path')
    parser.add_argument('-d','--datapath', default='./data',help='data path')
    args = parser.parse_args()
    fixed_seed(30678)

    checkpoint = args.checkpoint
    data_path = args.datapath
    batch_size = 32
    num_workers = 0
    train_loader, valid_loader, test_loader = preprocessing(data_path, batch_size, num_workers)

    if args.mode == 'train':
        train(train_loader, valid_loader, checkpoint)
        # val(checkpoint, valid_loader, batch_size)

    elif args.mode == 'test':
        test(checkpoint, test_loader)
