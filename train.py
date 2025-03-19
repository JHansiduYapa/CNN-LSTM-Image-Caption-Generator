# images: (batch_size, channels, height, width)
# captions: (batch_size, caption_length)
for epoch in range(num_epochs):
    for batch_idx, (images, captions) in enumerate(dataloader):
        images = images.to(device)
        captions = captions.to(device)
        
        optimizer.zero_grad()
        
        # forward pass through Encoder and Decoder
        features = encoder(images)  # shape: (batch_size, embed_size)
        outputs = decoder(features, captions) 
        
        
        # reshape outputs and targets for the loss
        # outputs: (batch_size*caption_length, vocab_size)
        # targets: (batch_size*caption_length)
        loss = criterion(outputs.reshape(-1, vocab_size), captions.reshape(-1))

        # add the loss values to a list
        loss_values.append(loss.item())
        
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}], Loss: {loss.item():.4f}")