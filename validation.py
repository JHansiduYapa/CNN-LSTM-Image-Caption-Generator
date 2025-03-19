val_dataloader = DataLoader(val_dataset, batch_size=1, collate_fn=lambda batch: collate_fn(batch, vocab))

encoder.eval()
decoder.eval()  

# disable the gradient calculation 
with torch.no_grad():
    i = 0
    for batch_idx, (images, captions) in enumerate(val_dataloader):
        # show the images with caption 
        image_cpu = images[0].to("cpu")
        image_np = image_cpu.numpy().transpose(1, 2, 0)
        plt.imshow(image_np)
        plt.axis("off")
        plt.show()

        # send to the same device
        images = images.to(device)
        captions = captions.to(device)
        
        # forward pass
        features = encoder(images)  # shape: (batch_size, embed_size)
        outputs = decoder.generate(features) 
        
        # convert to words
        predicted_captions = vocab.decode(outputs)
        correct_captions = vocab.decode(captions.squeeze(0).tolist())

        print("Correct: ", " ".join(correct_captions))
        print("Predicted: ", " ".join(predicted_captions))
        print("--------------------------------------------")

        if i == 100:
            break
        i += 1