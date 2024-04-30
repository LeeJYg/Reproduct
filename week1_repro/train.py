import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from Layer import RETRO
from transformers import get_linear_schedule_with_warmup

import os
import json

char_vocab = {'a': 1,
  'b': 2,
  'c': 3,
  'd': 4,
  'e': 5,
  'f': 6,
  'g': 7,
  'h': 8,
  'i': 9,
  'j': 10,
  'k': 11,
  'l': 12,
  'm': 13,
  'n': 14,
  'o': 15,
  'p': 16,
  'q': 17,
  'r': 18,
  's': 19,
  't': 20,
  'u': 21,
  'v': 22,
  'w': 23,
  'x': 24,
  'y': 25,
  'z': 26,
  'A': 27,
  'B': 28,
  'C': 29,
  'D': 30,
  'E': 31,
  'F': 32,
  'G': 33,
  'H': 34,
  'I': 35,
  'J': 36,
  'K': 37,
  'L': 38,
  'M': 39,
  'N': 40,
  'O': 41,
  'P': 42,
  'Q': 43,
  'R': 44,
  'S': 45,
  'T': 46,
  'U': 47,
  'V': 48,
  'W': 49,
  'X': 50,
  'Y': 51,
  'Z': 52,
  '0': 53,
  '1': 54,
  '2': 55,
  '3': 56,
  '4': 57,
  '5': 58,
  '6': 59,
  '7': 60,
  '8': 61,
  '9': 62,
  '!': 63,
  '"': 64,
  '#': 65,
  '$': 66,
  '%': 67,
  '&': 68,
  "'": 69,
  '(': 70,
  ')': 71,
  '*': 72,
  '+': 73,
  ',': 74,
  '-': 75,
  '.': 76,
  '/': 77,
  ':': 78,
  ';': 79,
  '<': 80,
  '=': 81,
  '>': 82,
  '?': 83,
  '@': 84,
  '[': 85,
  '\\': 86,
  ']': 87,
  '^': 88,
  '_': 89,
  '`': 90,
  '{': 91,
  '|': 92,
  '}': 93,
  '~': 94,
  '\n': 95,
  ' ' : 96,
  #'<sos>': 97,
  #'<eos>': 98,
  '<pad>': 0}
char_to_index = dict((char, index) for index, char in enumerate(char_vocab))
index_to_char = dict((index, char) for index, char in enumerate(char_vocab))

loss_file = 'training_loss_records.txt'

def write_loss_to_file(filename, epoch, loss, mode='train'):
    with open(filename, 'a') as file:
        file.write(f'{mode} Epoch: {epoch+1}, Loss: {loss:.4f}\n')
        
def load_and_preprocess_data():
    #data_path = os.path.join(os.getcwd(), '/home/jylee/week1_repro/data/retro_train_dataset.json')
    data_path = '/home/jylee/Reproduct/week1_repro/data/retro_train_dataset.json'
    with open(data_path, 'r') as f:
        dataset = json.load(f)

    train_dataset = dataset[:int(len(dataset) * 0.8)]
    val_dataset = dataset[int(len(dataset) * 0.8):]
     
    return train_dataset, val_dataset

def pad_retrieved_passages(retrieved_passages_list, pad_token_index=0, max_length=16):
    padded_list = []
    for i, passages in enumerate(retrieved_passages_list):
        for j, retrieved in enumerate(passages):
            for k, passage in enumerate(retrieved):
                if len(passage) < max_length:
                    pad_part = torch.full((max_length - len(passage),), pad_token_index, dtype=torch.long)
                    retrieved[i][j][k] = torch.cat((passage, pad_part), dim=0)
    return padded_list

def process_retrieved_passages(retrieved_passages_list, char_to_index, device):
    processed_retrieved_passages = []

    for passages in retrieved_passages_list:
        passages_tokenized = [[[char_to_index[char] for char in passage] for passage in retrieved] for retrieved in passages]
        processed_retrieved_passages.append(passages_tokenized)

    retrieved_passages_tensor = torch.tensor(processed_retrieved_passages).to(device)
    #retrieved_passages_tensor = pad_retrieved_passages(retrieved_passages_tensor)
    
    return retrieved_passages_tensor

def custom_collate_fn(batch, device):
    srcs, trgs, retrieved_passages_list = zip(*batch)
    
    srcs_tokenized = [torch.tensor([char_to_index[char] for char in src], dtype=torch.long) for src in srcs]
    trgs_tokenized = [torch.tensor([char_to_index[char] for char in trg], dtype=torch.long) for trg in trgs]
    
    retrieved_passages_tensor = process_retrieved_passages(retrieved_passages_list, char_to_index, device)

    srcs_padded = torch.nn.utils.rnn.pad_sequence(srcs_tokenized, batch_first=True, padding_value=char_to_index['<pad>']).to(device)
    trgs_padded = torch.nn.utils.rnn.pad_sequence(trgs_tokenized, batch_first=True, padding_value=char_to_index['<pad>']).to(device)

    return srcs_padded, trgs_padded, retrieved_passages_tensor


def train(model, train_loader, optimizer, criterion, device, epoch, writer, scheduler):
    model.train()
    total_loss = 0
    for idx, (src, trg, retrieved_passages) in enumerate(train_loader):
        src, trg, retrieved_passages = src.to(device), trg.to(device), retrieved_passages.to(device)
        #src는 32, 512/ trg도 32, 512/ retrieve_passages는 32, 32, 2, 32
   
        batch_size = src.shape[0]
        
        src = src.view(batch_size, 32,-1)
        trg = trg.view(batch_size, 32,-1)
        
        optimizer.zero_grad()
        output = model(src, trg, retrieved_passages)
        output_dim = output.shape[-1]
        
        output = output.contiguous().view(-1, output_dim)
        trg = trg.contiguous().view(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(train_loader)
    writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + idx)
    write_loss_to_file(loss_file, epoch, avg_loss, mode='train')

    return avg_loss

def evaluate(model, val_loader, criterion, device, epoch, writer):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for idx, (src, trg, retrieved_passages) in enumerate(val_loader):
            src, trg, retrieved_passages = src.to(device), trg.to(device), retrieved_passages.to(device)
            batch_size = src.shape[0]

            src = src.view(batch_size, 32,-1)
            trg = trg.view(batch_size, 32,-1)

            output = model(src, trg, retrieved_passages)
            output_dim = output.shape[-1]
            
            output = output.contiguous().view(-1, output_dim)
            trg = trg.contiguous().view(-1)
            
            loss = criterion(output, trg)
            total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)    
        writer.add_scalar('Loss/val', loss.item(), epoch * len(val_loader) + idx)
        write_loss_to_file(loss_file, epoch, avg_loss, mode='train')
        
    return avg_loss

def main():
    # 하이퍼파라미터 설정
    BATCH_SIZE = 64
    INIT_LR = 1e-7
    MAX_LR = 2e-4
    MIN_LR = 2e-5
    WEIGHT_DECAY = 0.1
    WARMUP_STEPS = 750
    NUM_STEP = 35000
    
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    writer = SummaryWriter()
    
    model = RETRO()
    model.to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-7, eps=1e-8, weight_decay=WEIGHT_DECAY)
    
    def lr_lambda(current_step: int):
        if current_step < WARMUP_STEPS:
            # 0 스텝에서 INIT_LR에서 시작하여 WARMUP_STEPS에서 MAX_LR에 도달하도록 선형 증가
            return (MAX_LR - INIT_LR) / WARMUP_STEPS * current_step + INIT_LR
        elif current_step < NUM_STEP:
            # WARMUP_STEPS에서 TOTAL_STEPS까지 MAX_LR에서 MIN_LR로 선형 감소
            return (MIN_LR - MAX_LR) / (NUM_STEP - WARMUP_STEPS) * (current_step - WARMUP_STEPS) + MAX_LR
        else:
            # TOTAL_STEPS 이후에는 MIN_LR 유지
            return MIN_LR
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    train_dataset, val_dataset= load_and_preprocess_data()
    steps_per_epoch = len(train_dataset) // BATCH_SIZE
    NUM_EPOCHS = NUM_STEP // steps_per_epoch
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=partial(custom_collate_fn, device=DEVICE), drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=partial(custom_collate_fn, device=DEVICE), drop_last=True)

    for epoch in range(NUM_EPOCHS):
        print("Training start!")
        print("Epoch: ", epoch+1, "/", NUM_EPOCHS)
        train_loss = train(model, train_loader, optimizer, criterion, DEVICE, epoch, writer, scheduler)
        scheduler.step()
        
        val_loss = evaluate(model, val_loader, criterion, DEVICE, epoch, writer)

        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if epoch % 20 == 0:
            torch.save(model.state_dict(), f'retro_model_epoch_{epoch}.pth')

    # 모델 저장
    torch.save(model.state_dict(), 'retro_model.pth')
    writer.close()

    print("Training complete.")
    
if __name__ == "__main__":
    main()    