import torch
from model import SphereFace

def verify(test_loader, threshold=0.5):
    # load the trained model
    model = SphereFace()
    model.load_state_dict(torch.load('./save/sphereface_model.pth'))

    correct = 0
    total = 0

    for i, (img1, img2, same) in enumerate(test_loader):
        # label = torch.tensor([0 if s else 1 for s in same])
        # label = torch.tensor([1] * len(same))
        output1 = model(img1)
        output2 = model(img2)
        # print(output1)
        # print(output2
        distance = torch.cosine_similarity(output1, output2) > threshold
        correct += (same == distance).sum().item()
        total += len(same)

    print("Correct number: {}, total number: {}".format(correct, total))
    print("The test accuracy is {:.4f}".format(correct / total))

    # transform = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor()])
    # image1 = transform(Image.open(image1_path)).unsqueeze(0)
    # image2 = transform(Image.open(image2_path)).unsqueeze(0)
    # label = torch.tensor([1])
    # output1 = model(image1, label)
    # output2 = model(image2, label)
    # distance = 1 - torch.cosine_similarity(output1, output2)
    # if distance < threshold:
    #     print(f"Verification result: SAME (distance: {distance:.4f})")
    # else:
    #     print(f"Verification result: DIFFERENT (distance: {distance:.4f})")

# test the verification function
# verify('image1.jpg', 'image2.jpg')

'''
Here is an example of how to use the verify function to perform face verification on a set of images:
# define a list of image pairs to verify
image_pairs = [('image1.jpg', 'image2.jpg'), ('image1.jpg', 'image3.jpg'), ('image2.jpg', 'image3.jpg')]

# loop over the image pairs and perform verification
for image1_path, image2_path in image_pairs:
    verify(image1_path, image2_path)
'''