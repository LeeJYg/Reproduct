import torch
from Layer import RETRO 
from train import char_to_index, index_to_char

torch.cuda.empty_cache()
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

#model = RETRO(embed_size=896, heads=16, enc_num_layers=12)
model = RETRO(device=device)
model = model.to(device)

# 가중치를 불러옵니다
model.load_state_dict(torch.load('/home/jylee/Reproduct/week1_repro/retro_model_epoch_1440.pth'))

# 모델을 추론 모드로 설정합니다
model.eval()

def tensor_to_sentence(output_tensor, index_to_char):
    # output_tensor 크기: (1, 2, 16, 97)
    # 배치 차원과 청크 차원을 합칩니다. 새로운 크기: (32, 97)
    output_tensor = output_tensor.view(-1, output_tensor.size(-1))
    
    # 각 위치에서 가장 확률이 높은 인덱스를 선택합니다.
    _, max_indices = torch.max(output_tensor, dim=1)
    
    # 인덱스를 글자로 변환합니다.
    predicted_chars = [index_to_char[index.item()] for index in max_indices]
    
    # 글자들을 연결하여 문장을 구성합니다.
    sentence = ','.join(predicted_chars)
    
    return sentence

def prepare_input_for_inference(input_text, char_to_index, chunk_len=16, device=device):
    # 텍스트를 인덱스 배열로 변환
    indexed_input = [char_to_index[char] for char in input_text]

    # 청크 길이에 맞게 텍스트를 패딩
    padding_length = chunk_len - (len(indexed_input) % chunk_len)
    indexed_input = [char_to_index['<pad>']] * padding_length + indexed_input # '<pad>'는 패딩 토큰

    # 입력 데이터를 청크로 분할
    chunks = [indexed_input[i:i+chunk_len] for i in range(0, len(indexed_input), chunk_len)]
    
    # 3차원 입력 형태로 변환: (batch size=1, chunk num, chunk len)
    input_tensor = torch.tensor(chunks).unsqueeze(0)
    
    return input_tensor.to(device)

def prepare_retrieved_passages(input_chunk_num, k, relevant_chunks, char_to_index, chunk_len, device=device):
    # batch size, input chunk num, k, relevant chunk의 형태로 변환
    retrieved_passages_tensor = torch.full((1, input_chunk_num, k, chunk_len), char_to_index['<pad>'], dtype=torch.long)
    
    for i, chunks in enumerate(relevant_chunks):
        for j, chunk in enumerate(chunks):
            # 청크를 인덱스 배열로 변환하고 왼쪽 패딩 추가
            indexed_chunk = [char_to_index[char] for char in chunk]
            padding_length = chunk_len - len(indexed_chunk)
            indexed_chunk = [char_to_index['<pad>']] * padding_length + indexed_chunk  # 왼쪽 패딩 추가
            retrieved_passages_tensor[0, i, j, :] = torch.tensor(indexed_chunk[:chunk_len], dtype=torch.long)
    
    return retrieved_passages_tensor.to(device)

def generate_text(model, input_sentence, relevant_chunks, max_length, device):
    trg_indexes = []
    for i in range(max_length):
        input_tensor = prepare_input_for_inference(input_sentence, char_to_index)
        
        input_chunk_num, chunk_len = input_tensor.shape[-2], input_tensor.shape[-1]
        retrieved_passages = prepare_retrieved_passages(input_chunk_num, 2, relevant_chunks, char_to_index, chunk_len, device)
        # 모델을 통해 다음 출력 토큰 예측
        with torch.no_grad():
            output = model(input_tensor, input_tensor, retrieved_passages)
        # 가장 최근에 예측된 토큰을 가져옴
        next_token = tensor_to_sentence(output, index_to_char)
        trg_indexes.append(next_token.split(',')[-1])
        input_sentence += trg_indexes[-1]
        print(input_sentence)

    # 생성된 텍스트의 인덱스를 실제 문자로 변환
    generated_text = trg_indexes
    
    return generated_text

input_text = "The weather today is"

relevant_chunks = [["sunny and warm.", "perfect for a walk."]]  # 관련된 정보 예시

generated_text = generate_text(model, input_text, relevant_chunks, max_length=64, device=model.device)

print(generated_text)