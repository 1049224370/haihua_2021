import argparse
import json
import os
from datetime import datetime
import pickle
import math as math
import torch
import transformers
from torch.nn import DataParallel
from torch.utils.data import Dataset
# from torch.utils.tensorboard import SummaryWriter

from modelMy import modelMy
from tokenizations.bpe_tokenizer import get_encoder


def build_files(data_path, full_tokenizer):
    if(os.path.exists('data/processed/')):
        f = open('data/processed/input_question_list.data', 'rb')
        # 将文件中的变量加载到当前工作区
        input_question_list = pickle.load(f)
        f.close()
        f = open('data/processed/resources_id.data', 'rb')
        # 将文件中的变量加载到当前工作区
        resources_id = pickle.load(f)
        f.close()
        f = open('data/processed/max_q_len.data', 'rb')
        # 将文件中的变量加载到当前工作区
        max_q_len = pickle.load(f)
        f.close()
        with open(data_path, 'r', encoding='utf8') as f:
            print('reading lines')
            lines = json.load(f)
            resources = [line['Content'] for line in lines]  # 用[SEP]表示换行, 段落之间使用SEP表示段落结束
        return resources,resources_id,input_question_list,max_q_len
    else:
        with open(data_path, 'r', encoding='utf8') as f:
            print('reading lines')
            lines = json.load(f)
            resources = [line['Content'] for line in lines]  # 用[SEP]表示换行, 段落之间使用SEP表示段落结束
            questions = [line['Questions'] for line in lines]

        n_resources = len(resources)
        input_question_list = []
        max_q_len = 0

        for i in range(n_resources):

            all_question_input = []
            for question in questions[i]:
                answer = (ord)(question['Answer'][0]) - (ord)('A')
                temp = {}
                temp['Question'] = question['Question']
                temp['Question_token'] = [full_tokenizer.convert_tokens_to_ids('[CLS]')]
                temp['Question_token'] = temp['Question_token'] + [full_tokenizer.convert_tokens_to_ids(word) for word in temp['Question']]
                temp['Question_token'] = temp['Question_token'] + [full_tokenizer.convert_tokens_to_ids('[SEP]')]
                temp['Question_token'] = temp['Question_token'] + full_tokenizer.convert_tokens_to_ids(['[PAD]']) * (512 - len(temp['Question_token']))
                temp['Choices'] = [] * 4
                temp['Goal'] = answer
                temp['Choices_token'] = [] * 4
                for j in range(4):
                    if(j >= len(question['Choices'])):
                        choice = ""
                    else:
                        choice = question['Choices'][j].replace("A.","").replace("B.","").replace("C.","").replace("D.","").replace("A．","").replace("B．","").replace("C．","").replace("D．","")
                    temp['Choices'].append(choice)
                    temp['Choices_token'].append([full_tokenizer.convert_tokens_to_ids('[CLS]')])
                    temp['Choices_token'][j] = temp['Choices_token'][j] + [full_tokenizer.convert_tokens_to_ids(word) for word in temp['Choices'][j]]
                    temp['Choices_token'][j] = temp['Choices_token'][j] + [full_tokenizer.convert_tokens_to_ids('[SEP]')]
                    temp['Choices_token'][j] = temp['Choices_token'][j] + full_tokenizer.convert_tokens_to_ids(['[PAD]']) * (256 - len(temp['Choices_token'][j]))
                    max_q_len = max(len(temp['Choices_token'][j]),max_q_len)
                all_question_input.append(temp)
            input_question_list.append(all_question_input)

        resources_token = [full_tokenizer.tokenize(line) for line in resources]
        resources_id = [full_tokenizer.convert_tokens_to_ids(line) for line in resources_token]
        os.mkdir('data/processed')
        f = open('data/processed/input_question_list.data', 'wb')
        # 将文件中的变量加载到当前工作区
        pickle.dump(input_question_list, f)
        f.close()
        f = open('data/processed/resources_id.data', 'wb')
        # 将文件中的变量加载到当前工作区
        pickle.dump(resources_id, f)
        f.close()
        f = open('data/processed/max_q_len.data', 'wb')
        # 将文件中的变量加载到当前工作区
        pickle.dump(max_q_len, f)
        f.close()
        return resources,resources_id,input_question_list,max_q_len

def sliding_window(max_len,resources,stride = None):
    window_len = max_len - 2
    if stride is None:
        stride = int(window_len / 2)
    ids = []
    token_type_ids = []
    for i in range(max(math.ceil((len(resources)-window_len)/stride),0) + 1):
        if(window_len+i*stride > len(resources)) :
            temp = [101] + resources[i * stride:] + [102]
            temp = temp + (max_len- len(temp)) * [0]
        else:
            temp = [101] + resources[i*stride:window_len+i*stride] + [102] # 102 is [SEP]
        ids.append(temp)
        token_type_ids.append([0 for z in range(max_len)])
    return ids,token_type_ids

class my_dataset(Dataset):
    def __init__(self, x, y, token_type_ids):
        self.x = x
        self.y = y
        self.token_type_ids = token_type_ids
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        text = self.x[idx]
        label = self.y[idx]
        token_type_ids = self.token_type_ids[idx]
        sample = {"ids": text, "label": label, "toen_type_ids": token_type_ids}
        return sample

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--model_config', default='gpt2/config.json', type=str, required=False,
                        help='选择模型参数')
    parser.add_argument('--tokenizer_path', default='cache/vocab_small.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--raw_data_path', default='data/train.json', type=str, required=False, help='原始训练语料')
    parser.add_argument('--tokenized_data_path', default='data/tokenized/', type=str, required=False,
                        help='tokenized语料存放位置')
    parser.add_argument('--raw', action='store_true', help='是否先做tokenize')
    parser.add_argument('--epochs', default=100, type=int, required=False, help='训练循环')
    parser.add_argument('--batch_size', default=1, type=int, required=False, help='训练batch size')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--warmup_steps', default=10000, type=int, required=False, help='warm up步数')
    # parser.add_argument('--log_step', default=2, type=int, required=False, help='多少步汇报一次loss，设置为gradient accumulation的整数倍')
    parser.add_argument('--stride', default=768, type=int, required=False, help='训练时取训练数据的窗口步长')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')
    parser.add_argument('--fp16', action='store_true', help='混合精度')
    parser.add_argument('--fp16_opt_level', default='O1', type=str, required=False)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--num_pieces', default=100, type=int, required=False, help='将训练语料分成多少份')
    parser.add_argument('--min_length', default=1, type=int, required=False, help='最短收录文章长度')
    parser.add_argument('--output_dir', default='model_classfier/', type=str, required=False, help='模型输出路径')
    parser.add_argument('--pretrained_model', default='', type=str, required=False, help='模型训练起点路径')
    parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, required=False, help='Tensorboard路径')
    parser.add_argument('--segment', action='store_true', help='中文以词为单位')
    parser.add_argument('--bpe_token', action='store_true', help='subword')
    parser.add_argument('--encoder_json', default="tokenizations/encoder.json", type=str, help="encoder.json")
    parser.add_argument('--vocab_bpe', default="tokenizations/vocab.bpe", type=str, help="vocab.bpe")

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    if args.segment:
        from tokenizations import tokenization_bert_word_level as tokenization_bert
    else:
        from tokenizations import tokenization_bert

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡

    model_config = transformers.GPT2Config.from_json_file(args.model_config)
    print('config:\n' + model_config.to_json_string())

    n_ctx = model_config.n_ctx
    if args.bpe_token:
        full_tokenizer = get_encoder(args.encoder_json, args.vocab_bpe)
    else:
        full_tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
    full_tokenizer.max_len = 999999
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('using device:', device)

    raw_data_path = args.raw_data_path
    # tokenized_data_path = args.tokenized_data_path
    raw = args.raw  # 选择是否从零开始构建数据集
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    warmup_steps = args.warmup_steps
    # log_step = args.log_step
    # stride = args.stride
    gradient_accumulation = args.gradient_accumulation
    # fp16 = args.fp16  # 不支持半精度的显卡请勿打开
    # fp16_opt_level = args.fp16_opt_level
    max_grad_norm = args.max_grad_norm
    # num_pieces = args.num_pieces
    # min_length = args.min_length
    output_dir = args.output_dir
    # tb_writer = SummaryWriter(log_dir=args.writer_dir)
    # assert log_step % gradient_accumulation == 0

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if raw:
        print('building files')
        resources,resources_id,input_question_list, max_aq_len = build_files(data_path=raw_data_path, full_tokenizer=full_tokenizer)
        print('files built')
    input_ids = [] * len(resources_id)
    # labels = []
    for i in range(len(resources_id)):
        inputsss,_ = sliding_window( max_len = 512, resources = resources_id[i], stride = 512-128)
        input_ids.append(inputsss)
        # labels = labels + [choices['label']] * len(inputsss)
    print('sliding built')
    a=1
    val_rate = 0.1
    split = int((1-val_rate) * len(input_ids))
    val_input_ids = input_ids[split:]
    val_input_question_list = input_question_list[split:]

    input_ids = input_ids[:split]
    input_question_list = input_question_list[:split]

    # if False:  # shuffle
    #     index = [i for i in range(len(token_type_ids))]
    #     random.shuffle(index)
    #     new_input_ids = [input_ids[i] for i in index]
    #     new_labels = [labels[i] for i in index]
    #     new_token_type_ids = [token_type_ids[i] for i in index]
    #     input_ids = new_input_ids
    #     labels = new_labels
    #     token_type_ids = new_token_type_ids
    # train_dataset = my_dataset(x=input_ids, y=labels, token_type_ids=token_type_ids)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    # if not args.pretrained_model:
    #     model = transformers.models.gpt2.GPT2LMHeadModel(config=model_config)
    # else:
    #     model = transformers.models.gpt2.GPT2LMHeadModel.from_pretrained(args.pretrained_model)

    model = modelMy(args,device)

    model.to(device)

    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    print('number of parameters: {}'.format(num_parameters))

    multi_gpu = False
    print('calculating total steps')
    # for i in tqdm(range(num_pieces)):
    #     with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'r') as f:
    #         full_len += len([int(item) for item in f.read().strip().split()])

    optimizer = transformers.optimization.AdamW(model.parameters(), lr=lr, correct_bias=True)
    scheduler = transformers.optimization.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps = args.epochs * len(resources))
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = DataParallel(model, device_ids=[int(i) for i in args.device.split(',')])
        multi_gpu = True
    print('starting training')
    overall_step = 0
    running_loss = 0
    best_loss = 9999999
    for epoch in range(epochs):
        print('epoch {}'.format(epoch + 1))
        now = datetime.now()
        print('time: {}'.format(now))
        # x = np.linspace(0, num_pieces - 1, num_pieces, dtype=np.int32)
        acc_s = 0
        piece_num = 0

        model.train()
        for step in range(len(input_ids)):  # paper by paper
            # print ("step:{}".format(step))
            if (step%1000 == 0):
                print("step={}".format(step))
            batch_inputs = input_ids[step]
            batch_inputs = torch.tensor(batch_inputs).long().to(device).unsqueeze(0)
            batch_questions = [z['Question_token'] for z in input_question_list[step][:]]
            batch_questions = torch.tensor(batch_questions).long().to(device).unsqueeze(0)
            batch_choices = [z['Choices_token'] for z in input_question_list[step][:]]
            batch_choices = torch.tensor(batch_choices).long().to(device).unsqueeze(0)
            batch_labels = [z['Goal'] for z in input_question_list[step][:]]
            batch_labels = torch.tensor(batch_labels).long().to(device).unsqueeze(0)
            #  forward pass
            outputs = model.forward(inputs=batch_inputs,questions = batch_questions, choices = batch_choices, labels=batch_labels)
            loss, pred, acc = outputs
            acc_s += (acc.cpu())
            running_loss += loss.item()
            #  get loss
            if multi_gpu:
                loss = loss.mean()
            if gradient_accumulation > 1:
                loss = loss / gradient_accumulation

            #  loss backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            #  optimizer step
            if (overall_step + 1) % gradient_accumulation == 0:

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            # if (overall_step + 1) % log_step == 0:
            #     tb_writer.add_scalar('loss', loss.item() * gradient_accumulation, overall_step)
            overall_step += 1
        piece_num += 1
        running_loss = running_loss / len(input_ids)
        print('now time: {}:{}. epoch {}, loss {}, acc {:.6f}'.format(
            datetime.now().hour,
            datetime.now().minute,
            epoch + 1,
            running_loss,
            acc_s / len(input_ids)
            # acc_s / 800
            ))
        # running_loss = running_loss * gradient_accumulation / len(resources)
        if running_loss < best_loss:
            best_loss = running_loss
            print('saving model for epoch {}'.format(epoch + 1))
            if not os.path.exists(output_dir + 'model_epoch{}'.format(epoch + 1)):
                os.mkdir(output_dir + 'model_epoch{}'.format(epoch + 1))
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(output_dir + 'loss.best', optimizer, epoch)
        running_loss = 0

        model.eval()
        val_accs = 0
        for stepp in range(len(val_input_ids)):
            if True and (stepp>200):
                break
            # input_ids = []
            # labels = []
            # token_type_ids = []
            batch_inputs = val_input_ids[stepp]
            batch_inputs = torch.tensor(batch_inputs).long().to(device).unsqueeze(0)
            batch_questions = [z['Question_token'] for z in val_input_question_list[stepp][:]]
            batch_questions = torch.tensor(batch_questions).long().to(device).unsqueeze(0)
            batch_choices = [z['Choices_token'] for z in val_input_question_list[stepp][:]]
            batch_choices = torch.tensor(batch_choices).long().to(device).unsqueeze(0)
            batch_labels = [z['Goal'] for z in val_input_question_list[stepp][:]]
            batch_labels = torch.tensor(batch_labels).long().to(device).unsqueeze(0)
            #  forward pass
            outputs = model.forward(inputs=batch_inputs, questions=batch_questions, choices=batch_choices,
                                    labels=batch_labels,training=False)
            loss, pred, acc = outputs
            val_accs += (acc)
            #  get loss
            #  loss backward
            #  optimizer step
        # torch.save(scheduler.state_dict(), output_dir + 'model_epoch{}/scheduler.pt'.format(epoch + 1))
        # torch.save(optimizer.state_dict(), output_dir + 'model_epoch{}/optimizer.pt'.format(epoch + 1))
        # val_accs = val_accs/len(val_input_ids)
        # val_accs = val_accs / 70
        print('validation acc {}'.format(val_accs))
        print('epoch {} finished'.format(epoch + 1))

        then = datetime.now()
        print('time: {}'.format(then))
        print('time for one epoch: {}'.format(then - now))

    print('training finished')
    if not os.path.exists(output_dir + 'final_model'):
        os.mkdir(output_dir + 'final_model')
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir + 'final.best', optimizer, epoch)
    # torch.save(scheduler.state_dict(), output_dir + 'final_model/scheduler.pt')
    # torch.save(optimizer.state_dict(), output_dir + 'final_model/optimizer.pt')


if __name__ == '__main__':
    main()