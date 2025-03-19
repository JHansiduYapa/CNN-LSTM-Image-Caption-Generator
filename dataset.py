# extend the dataset class from pytorch to get the data
class FlickrDataset(Dataset):
    def __init__(self, captions_file, images_dir, transform=None, vocab=None):
        # captions_file: path to captions.txt file.
        # images_dir: directory with images.
        # transform: torchvision transforms for image pre-processing.
        # vocab: Vocabulary object.
        self.images_dir = images_dir
        self.transform = transform
        self.vocab = vocab
        # list of tuples: (image_filename, caption)
        self.image_captions = []

        # open the file and fill the image_captions list with
        with open(captions_file, 'r') as f:
            for i, line in enumerate(f):
                # skip the first line header
                if i == 0:
                    continue 
                line = line.strip()
                if line:
                    # split on the first comma only
                    image_name, caption = line.split(',', 1)
                    all_captions.append(caption)
                    # add to the image_captions list
                    self.image_captions.append((image_name, caption))

    # get the length of the dataset 
    def __len__(self):
        return len(self.image_captions)

    # define how to get an item from dataset
    def __getitem__(self, idx):
        # from given index get an image_caption dataset
        img_name, caption = self.image_captions[idx]
        # add to the image directory
        img_path = os.path.join(self.images_dir, img_name)
        # open the image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # numericalize caption with start and end tokens
        tokens = [self.vocab.stoi["<start>"]]
        tokens += self.vocab.encode(caption)
        tokens.append(self.vocab.stoi["<end>"])
        caption_tensor = torch.tensor(tokens)
        # return image and the caption tensor
        return image, caption_tensor