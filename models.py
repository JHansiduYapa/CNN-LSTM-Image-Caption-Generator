## with the pre-trained model
class Encoder(nn.Module):
    def __init__(self, embed_size):
        super(Encoder, self).__init__()
        # Load the pretrained ResNet-101 model
        resnet = models.resnet101(pretrained=True)
        # Remove the last fully connected layer
        modules = list(resnet.children())[:-1]  # remove the original FC layer
        self.resnet = nn.Sequential(*modules)
        # New fully connected layer to map from resnet's output features to embed_size
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        # Optionally add a batch normalization layer
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        # Pass the images through ResNet-101 up to the last pooling layer.
        features = self.resnet(images)  # shape: (batch_size, 2048, 1, 1)
        features = features.view(features.size(0), -1)  # shape: (batch_size, 2048)
        features = self.fc(features)                  # shape: (batch_size, embed_size)
        features = self.bn(features)                  # shape: (batch_size, embed_size)
        return features
    
class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size

    def forward(self, features, captions):
        # features: encoded image features, shape (batch, embed_size)
        # captions: tokenized captions, shape (batch, caption_length)
        
        # add seq_len = 1 to the feature of the image
        features = features.unsqueeze(1)
        
        # for teacher forcing method we should shift right and add first time step as the encoding of the image
        embed_captions = self.embed(captions)

        # then add the initial state to the sequence by right shifting by one and add featues to 0 th position 
        embed_captions_shifted = embed_captions[:, :-1, :]

        # concatenate from the seq_len position
        inputs = torch.cat((features, embed_captions_shifted), dim=1)

        # then pass to the lstm
        # Defaults to zeros if (h_0, c_0) is not provided.
        lstm_out, _ = self.lstm(inputs)

        # then pass to the lenear layer to make it as vocabulary size
        outputs = self.linear(lstm_out)

        return outputs # (batch_size, seq_len, vocabulary_size)

    def generate(self, features, max_length=40):
        # normally batch_size == 1 for generation
        batch_size = features.size(0)
        
        hidden = (torch.zeros(self.lstm.num_layers, batch_size, self.hidden_size, device=features.device),
        torch.zeros(self.lstm.num_layers, batch_size, self.hidden_size, device=features.device))
        
        # The initial input is the image feature
        # Our LSTM is defined with batch_first=True, so we create a time dimension.
        inputs = features.unsqueeze(1)  # shape: (batch, 1, embed_size)
        
        generated_tokens = []
        
        for t in range(max_length):
            # Pass the current input through the LSTM.
            lstm_out, hidden = self.lstm(inputs, hidden)  # lstm_out: (batch, 1, hidden_size)
            
            # project LSTM output to vocabulary logits
            outputs = self.linear(lstm_out.squeeze(1))  # shape: (batch, vocab_size)
            
            # compute probabilities using softmax.
            probs = torch.softmax(outputs, dim=1)  # shape: (batch, vocab_size)
            
            # greedily select the word with the highest probability
            predicted = probs.argmax(dim=1)  # shape: (batch,)
            token_id = predicted.item()
            generated_tokens.append(token_id)
            
            # if the predicted token is the <end> token (assumed id = 2), stop generation
            if token_id == 2:
                break
            
            # prepare the next input by converting the predicted token into its embedding.
            inputs = self.embed(predicted).unsqueeze(1)  # shape: (batch, 1, embed_size)
        
        return generated_tokens