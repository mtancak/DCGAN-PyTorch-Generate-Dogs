import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    x = torch.tensor([1, 2, 3])
    print(x)

    # create instance of RN
    # create a CNN that takes latent noise and outputs an image
    # training loop
        # generate images from generator
        # run them through discriminator
        # after some iterations, switch
        # remember to freeze model at theright time using .train() and .inference()?