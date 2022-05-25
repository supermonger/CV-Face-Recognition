from train import train, test, preprocessing

if __name__ == "__main__":
    checkpoint = "best.pt"
    batch_size = 32
    train_loader, valid_loader = preprocessing(batch_size)
    train(train_loader, valid_loader)
    test(checkpoint, valid_loader, batch_size)