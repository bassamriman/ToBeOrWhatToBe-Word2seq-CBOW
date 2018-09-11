import argparse

import torch
import torch.nn as nn
from torchtext.data import BucketIterator
from torchtext.data import Dataset
from torchtext.data import Example
from torchtext.data import Field
from torchtext.data import TabularDataset


class CBOW(torch.nn.Module):

    def __init__(self, input_vocab_size, output_vocab_size, embedding_dim):
        super(CBOW, self).__init__()

        # out: 1 x emdedding_dim
        self.embeddings = nn.Embedding(input_vocab_size, embedding_dim)

        self.linear1 = nn.Linear(embedding_dim, 128)

        self.activation_function1 = nn.ReLU()

        # out: 1 x vocab_size
        self.linear2 = nn.Linear(128, output_vocab_size)

        self.activation_function2 = nn.LogSoftmax(dim=-1)

    def forward(self, inputs):
        out = sum(self.embeddings(inputs))
        out = self.linear1(out)
        out = self.activation_function1(out)
        out = self.linear2(out)
        out = self.activation_function2(out)
        return out


def train(log_interval, model, train_loader, optimizer, loss_function, epoch):
    for batch_idx, data in enumerate(train_loader):
        contexts, targets = data.sentence, data.verb_form

        optimizer.zero_grad()
        output = model(contexts)
        loss = loss_function(output, targets)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, test_loader, loss_function):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            contexts, targets = data.sentence, data.verb_form
            output = model(contexts)
            test_loss += loss_function(output, targets).item()
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))


def torch_text_from_csv():
    tokenize = lambda x: x.split()
    # Define columns in csv
    SENTENCE_FIELD = Field(sequential=True, tokenize=tokenize, pad_token="<unk>")
    VERB_FORM_FIELD = Field(sequential=False)
    datafields = [("sentence", SENTENCE_FIELD), ("verb_form", VERB_FORM_FIELD)]

    # Load and split data
    training_dataset, validation_dataset, test_dataset = TabularDataset(path="data.csv", format="csv", skip_header=True,
                                                                        fields=datafields).split([0.80, 0.10, 0.10])
    # Build vocabulary
    SENTENCE_FIELD.build_vocab(training_dataset)
    VERB_FORM_FIELD.build_vocab(training_dataset)
    return SENTENCE_FIELD, VERB_FORM_FIELD, training_dataset, validation_dataset, test_dataset

# used for troubleshooting
def torch_text_from_memory():
    tokenize = lambda x: x.split()
    SENTENCE_FIELD = Field(sequential=True, tokenize=tokenize, pad_token="<unk>")
    VERB_FORM_FIELD = Field(sequential=False)
    datafields = [("sentence", SENTENCE_FIELD), ("verb_form", VERB_FORM_FIELD)]

    # data = [{"sentence": "I king", "verb_form": "am"},
    #        {"sentence": "You my friend", "verb_form": "were"},
    #        {"sentence": "They my friend", "verb_form": "are"},
    #        {"sentence": "We kings", "verb_form": "are"},
    #        {"sentence": "I have strong", "verb_form": "been"},
    #        {"sentence": "We enemies", "verb_form": "were"}]
    data = [(
        "When the modern Olympics began in 1896, the initiators and organizers looking for a great popularizing event",
        "were"),
        ("I king", "am"),
        ("You my friend", "were"),
        ("They my friend", "are"),
        ("We kings", "are"),
        ("I have strong", "been"),
        ("We enemies", "were")]
    examples = []
    for d in data:
        examples.append(Example.fromlist(d, datafields))

    training_dataset, validation_dataset, test_dataset = Dataset(examples, datafields).split([0.33, 0.33, 0.33])
    SENTENCE_FIELD.build_vocab(training_dataset)
    VERB_FORM_FIELD.build_vocab(training_dataset)
    return SENTENCE_FIELD, VERB_FORM_FIELD, training_dataset, validation_dataset, test_dataset


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64 , metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--embedding-dim', type=int, default=100, metavar='N',
                        help='Embedding dimension')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    SENTENCE_FIELD, VERB_FORM_FIELD, training_dataset, validation_dataset, test_dataset = torch_text_from_csv()
    #SENTENCE_FIELD, VERB_FORM_FIELD, training_dataset, validation_dataset, test_dataset = torch_text_from_memory()

    print(training_dataset[0].sentence[:17])
    print(validation_dataset[0].sentence[:3])
    print(test_dataset[0].sentence[:3])

    print(len(training_dataset))
    print(len(validation_dataset))
    print(len(test_dataset))

    training_data_iterator, validation_date_iterator, test_data_iterator = BucketIterator.splits(
        (training_dataset, validation_dataset, test_dataset),
        batch_sizes=(args.batch_size, args.batch_size, args.batch_size),
        device=device,
        sort_key=lambda x: len(x.sentence),
        sort_within_batch=False,
        repeat=False
    )

    input_vocab_size = len(SENTENCE_FIELD.vocab)
    print("input_vocab_size " + str(input_vocab_size))
    output_vocab_size = len(VERB_FORM_FIELD.vocab)
    print("output_vocab_size " + str(output_vocab_size))

    model = CBOW(input_vocab_size, output_vocab_size, args.embedding_dim)
    model.to(device)

    loss_function = nn.NLLLoss()

    optimizer = torch.optim.SGD(model.parameters(), args.lr)

    for epoch in range(0, args.epochs):
        train(args.log_interval, model, training_data_iterator, optimizer, loss_function, epoch)
        test(model, test_data_iterator, loss_function)

    torch.save(model, "model")


if __name__ == '__main__':
    main()
