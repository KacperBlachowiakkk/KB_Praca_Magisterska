from torchvision import transforms

def build_train_transform(
    image_size: int = 224,
):

    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size)
        ),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(
                0.485,
                0.456,
                0.406,
            ),
            std=(
                0.229,
                0.224,
                0.225,
            ),
        ),
    ])


def build_eval_transform(
    image_size: int = 224,
):

    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(
                0.485,
                0.456,
                0.406,
            ),
            std=(
                0.229,
                0.224,
                0.225,
            ),
        ),
    ])