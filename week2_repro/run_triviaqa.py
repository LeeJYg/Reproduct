from inference import inference
import jsonlines

#1819246

loss_file = 'correct.txt'

def write_loss_to_file(filename, iter, total_data_num, correct, elem):
    with open(filename, 'a') as file:
        file.write(f"iter: {iter}/{total_data_num}, correct: {correct}, id: {elem['id']}\n")

def run():
    jsonl_file_path = '/home/jylee/Reproduct/week2_repro/triviaqa_test_w_gs.jsonl'
    
    data = []
    with jsonlines.open(jsonl_file_path) as reader:
        for text in reader:
           data.append(text)

    total_data_num = len(data)
    correct = 2318
    start_id = 'qw_7026'
    start = False
    iter_num = 3701
    
    for iter, elem in enumerate(data):  
        if iter == iter_num:
            start_id = elem['id']
    
    for iter, elem in enumerate(data):
        #start id부터 시작
        if elem['id'] == start_id:
            start = True
        if not start:
            continue
        
        answers = elem['answers']
        print(f"iter: {iter}/{total_data_num}", correct, elem['id'])
        write_loss_to_file(loss_file, iter, total_data_num, correct, elem)
        
        sentence, do_ret = inference(element=elem)
        
        if do_ret == 'Retrieved':
            sentence = sentence.split('</paragraph>')[1]
            for answer in answers:
                if answer in sentence:
                    correct += 1
                    break
        else:
            sentence = sentence.split('[No Retrieval]')[1]
            for answer in answers:
                if answer in sentence:
                    correct += 1
                    break
    
    print(f"Accuracy: {correct}/{total_data_num}")
    
if __name__ == '__main__':
    run()